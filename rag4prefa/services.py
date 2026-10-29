"""组件装配：所有依赖在这里创建一次，通过显式注入往下传。

没有模块级单例、没有可变全局状态——除了 SQLite 连接（单进程、WAL、串行写）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import repo
from .adapters.embed import Embedder, EmbeddingError
from .adapters.llm import LLM, LLMError
from .adapters.rerank import NoopReranker, RemoteReranker, Reranker
from .adapters.term_map import TermMap, load_term_map
from .adapters.vectors import (
    BackendUnavailable,
    MilvusStore,
    NumpyStore,
    VectorStore,
)
from .concurrency import MaintenanceBusy, maintenance_lock, worker_alive
from .config import Settings, get_settings
from .consistency import (
    IndexMismatch,
    collect_issues,
    ensure_active_backend,
    ensure_backend_current,
    inspect,
)
from .credentials import CredentialPool, UserCredentials
from .db import connect, init_db
from .ocr_setup import effective
from .repo import (
    get_meta,
    get_vector_backend,
    mark_interrupted_documents,
    mark_interrupted_jobs,
    set_vector_backend,
)


def _describe(exc: BaseException) -> str:
    """异常摘要。有些异常（裸 `RuntimeError()`）没有消息，那就只留类型名，
    否则 notes 里会出现一条以冒号结尾的空话。"""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


@dataclass
class Services:
    settings: Settings
    conn: sqlite3.Connection
    store: VectorStore
    embedder: Embedder | None
    llm: LLM | None
    reranker: Reranker
    notes: list[str] = field(default_factory=list)
    fell_back: bool = False
    # 本进程"确认过活动后端"的结论。三态，别合并：
    #   None = 从没跑过确认流程（CLI、库层面直接构造、测试夹具）；
    #   True = 确认过（登记成功，或者库里本来记着的就是手里这个）；
    #   False = 跑过但没能证明（抢不到锁、重建失败、读不到后端条数）。
    # 只有 False 会被检索挡下：那时本进程确实试过，并且知道自己没说服自己。
    backend_confirmed: bool | None = None
    # 多用户自带 Key：池子按指纹复用客户端，视图共享同一个池
    credentials: CredentialPool | None = None
    user_label: str | None = None
    # 这个视图是按哪份凭据拍出来的快照。`user_label` 只记用户名（任何人都非空，
    # 匿名也算 "anonymous"），判断"花谁的钱"必须看这里——见 `llm_billable`。
    user_creds: UserCredentials | None = None
    # 受控词表（公众表达 → 规范用语）。装配时读**一次**：每次问答都去解析一遍 JSON，
    # 等于把"文件改了一半"的风险摊到每个请求上。`None` = 未配置或加载失败，
    # 两者的区别在 `term_map_error` 里，不能都当成"没配"。
    term_map: TermMap | None = None
    term_map_error: str | None = None
    # 视图不拥有连接：一次请求结束若能关掉全局 SQLite 连接，下一个请求就全崩
    owns_conn: bool = True
    # 请求级视图指回根服务。后端指针（store / fell_back）只能由根服务改：
    # 视图是 `replace()` 出来的临时对象，请求结束就扔，在它上面迁移等于"迁移成功、
    # 根服务还在用旧后端"。见 `owner` 与 `maybe_promote`。
    root: Services | None = None

    def __post_init__(self) -> None:
        if self.credentials is None:
            self.credentials = CredentialPool(self.settings, self.conn)

    @property
    def owner(self) -> Services:
        """后端指针的所有者。视图返回根服务，根服务返回自己。"""
        return self.root if self.root is not None else self

    def for_credentials(self, creds: UserCredentials | None) -> Services:
        """换上某个用户 Key 之后的视图，共享 conn / store / reranker。

        没有凭据时原样返回 `self`，未登录访问继续用 `.env` 里的公共 Key——
        CLI、测试和已有部署的行为完全不变。
        """
        if creds is None or self.credentials is None:
            return self
        embedder, llm = self.credentials.acquire(creds)
        return replace(
            self,
            settings=self.credentials.settings_for(creds),
            embedder=embedder,
            llm=llm,
            notes=list(self.notes),
            user_label=creds.username,
            user_creds=creds,
            owns_conn=False,
            root=self.owner,
        )

    @property
    def llm_billable(self) -> bool:
        """这次请求的 LLM 调用是不是"允许发生"的。

        只有两种情况允许：用户自己带了 LLM Key，或者部署明确接受服务端代付
        （`R4P_ALLOW_SERVER_KEY_FALLBACK=true`，单机演示的默认值）。

        为什么不能用 `user_label` 判断（2026-10-27 复审 P1-1）：用户名对任何请求都非空
        ——只传一个模型头也会解析成 `anonymous`。拿它当"已登录"用，项目详情那条
        "生成摘要"的侧门就会替匿名请求调服务端 LLM；而这条路径不走 `svc_billable`，
        主入口拦得再严也堵不住它。
        """
        return bool(self.user_creds is not None and self.user_creds.llm_key) or (
            self.settings.allow_server_key_fallback
        )

    @property
    def index_dim(self) -> int | None:
        value = get_meta(self.conn, "embedding_dim")
        return int(value) if value else None

    @property
    def index_model(self) -> str | None:
        return get_meta(self.conn, "embedding_model")

    def require_embedder(self) -> Embedder:
        if not self.embedder:
            raise EmbeddingError("未配置 embedding 服务（R4P_EMBEDDING_API_KEY）")
        return self.embedder

    def require_llm(self) -> LLM:
        if not self.llm:
            raise LLMError("未配置大模型服务（R4P_LLM_API_KEY）")
        return self.llm

    def settle_backend(self) -> bool:
        """确认"这个库的活动后端"——**能把集合打开，不等于它就是当前版本**。

        为什么必须单独一步（2026-10-24 第十五轮复审）：`open_store()` 只要集合能打开
        就返回，而"能打开"证明不了"里面装的是归档这一套"。遗留的 Milvus 集合可以与
        归档条数相同、chunk_id 却完全不同（重建中断、换过一批文档）。此时把它登记为
        活动后端，后续写入就落进一个内容不对的索引——而条数一致时，连"索引 vs 归档"
        的计数校验都发现不了。

        协议只有一条：**先证明，再登记**（第十六轮把"证明"收紧到内容级）。

        - 同一后端且已登记 → 直接确认。这是启动的常规路径，不能每次整库重写；
        - 归档非空而登记值不是它（首次确认或换后端）→ **按归档整体重建**，逐 id 校验
          通过后才由 `rebuild_store` 在同一个事务里登记。只比 id 集合是不够的：id 相同
          而向量值、`doc_year`、`project_code` 是旧的那种索引，每天都在正常返回结果；
        - 归档为空 → 后端也必须读得出是空的。**读不到不是零**：SDK/连接抖一下就把
          "无法确认"降级成"确认为空"，等于让一个证明不了自己的后端进了登记表；
        - 证明不了 → 不登记，`backend_confirmed=False`，检索与写入都拒绝。

        整段在维护锁里：登记活动后端与"谁在写"必须互斥，否则刚证明完就被另一个进程
        改掉了。锁拿不到就跳过（不登记、也不确认），下次启动再核对——启动不该被别人的
        入库卡住，但也不能因此假装自己确认过。
        """
        owner = self.owner
        if owner is not self:
            return owner.settle_backend()
        recorded = get_vector_backend(self.conn)
        if recorded == self.store.name:
            # 快速路径也要有证据：库里记着的是 "milvus"，不等于**手里这个对象**已经打开。
            # `MilvusStore._open()` 抢不到维护锁时会 `_defer()`，对象照样名叫 milvus
            # （2026-10-24 第十七轮复审）。
            problem = self._prove_current()
            if problem is None:
                self.backend_confirmed = True
                return True
            self.backend_confirmed = False
            self.notes.append(problem)
            return False
        try:
            with maintenance_lock(
                self.conn, purpose=f"确认活动后端 {self.store.name}", timeout=0.0
            ):
                self.backend_confirmed = self._settle_locked(recorded)
        except MaintenanceBusy as exc:
            self.backend_confirmed = False
            self.notes.append(
                f"未确认活动后端（{exc}）——另一个维护动作正在改索引。"
                "本次按配置打开的后端不会登记，检索与写入都会拒绝；确认完成前请重启服务"
            )
        except Exception as exc:  # noqa: BLE001 - 确认失败不能让服务起不来
            self.backend_confirmed = False
            self.notes.append(
                f"确认活动后端时出错（{_describe(exc)}）；本次不登记，"
                "检索与写入都会拒绝。修好上面的问题后重启服务即可重试"
            )
        return bool(self.backend_confirmed)

    def _prove_current(self) -> str | None:
        """库里登记的就是这个名字时，还要证明**手里这份**能用、而且位置也对。

        返回 None 表示确认通过，否则返回一句可以直接给人看的原因。三件事：

        1. **对象是否真的打开了**。`MilvusStore` 在抢不到维护锁时会推迟打开，对象
           仍然是 "milvus"；把它当已确认，`/api/health` 会报正常，而查询要等到
           `search()` 里才发现后端不可用——那里通常只处理 `IndexMismatch`，于是表现为
           500 而不是 503；
        2. **是不是同一个存储位置**。类型相同、集合不同时，两个进程会各自写各自的
           集合，而所有基于类型名的检查都通过；
        3. 历史库没记过位置（`vector_backend_id` 为空）时**就地补记**，别为此重建一遍
           ——启动时间不该随库变大。
        """
        try:
            probe = getattr(self.store, "probe", None)
            if probe is not None:
                probe()
        except Exception as exc:  # noqa: BLE001 - 证明不了就是没证明
            return (
                f"库里登记的向量后端是「{self.store.name}」，但这个进程打开它时没能就绪"
                f"（{_describe(exc)}）。本次不确认：宁可暂时查不了，也不能拿一个没打开的"
                "索引去回答问题。等占着维护动作的进程结束后重启服务；"
                "若确认那个进程已不存在，直接重启即可"
            )

        mine = getattr(self.store, "identity", None) or None
        theirs = repo.get_vector_backend_id(self.conn)
        if mine and theirs and mine != theirs:
            return (
                f"库里登记的后端位置是「{theirs}」，本进程打开的是「{mine}」——"
                "后端类型相同，但不是同一份索引。共用同一个数据库的进程必须使用完全相同的 "
                "R4P_MILVUS_URI / R4P_MILVUS_COLLECTION；本次不登记也不确认，检索与写入都会"
                "拒绝。确认要用这一份，执行 `python -m rag4prefa rebuild` 从归档重建并登记"
            )
        if mine and not theirs:
            # 只补身份、不改登记的类型：这条记录可能在我们读完之后已经被别的进程改掉了
            repo.remember_backend_identity(self.conn, self.store.name, mine)
        return None

    def _settle_locked(self, recorded: str | None) -> bool:
        """在维护锁里确认后端。返回"是否已登记为活动后端"。"""
        target = self.store.name
        archive = repo.recoverable_chunk_ids(self.conn)
        if not archive:
            # 归档为空：没有内容可比对，只能核对"索引也必须是空的"。
            try:
                leftover = self.store.count()
            except Exception as exc:  # noqa: BLE001 - 读不到 ≠ 空
                # 以前这里把异常折成 0：SDK 超时、连接被占、Lite 文件锁住，全都被读成
                # "后端是空的"，接着登记。之后 `open_store` 又把它当"已有活动后端"，
                # 于是这个从没被证实过的后端一路被信任下去（2026-10-24 第十六轮复审）。
                self.notes.append(
                    f"归档为空，但读不到 {target} 索引的实际条数（{_describe(exc)}），"
                    "无法确认它是空的；本次不登记活动后端。"
                    "执行 `python -m rag4prefa doctor` 看后端到底怎么了"
                )
                return False
            if leftover:
                self.notes.append(
                    f"归档里没有任何片段，而 {target} 索引里还有 {leftover} 条向量"
                    "（上一版留下的）；未登记为活动后端。"
                    "执行 `python -m rag4prefa rebuild` 清掉这些残留后再启动"
                )
                return False
            set_vector_backend(
                self.conn, target, identity=getattr(self.store, "identity", None)
            )
            if recorded and recorded != target:
                self.notes.append(
                    f"活动后端由「{recorded}」切换为「{target}」（库为空，无需迁移）"
                )
            return True

        # 归档非空、登记值又不是它：**内容必须整份重写一遍才算证明**。
        #
        # 只比 id 集合是上一轮的做法，它证明的是"记录的身份一致"。Milvus 里还有向量值、
        # doc_year、project_code：从 Milvus 切到 numpy、在 numpy 上重建过向量或改过年份、
        # 旧集合一直没同步——这时的 id 集合可以完全相同，而内容整整差一代。日常校验
        # 只看条数/模型/维度，一条也发现不了。
        from .ingest import rebuild_store  # 局部导入，避免循环依赖

        try:
            moved = rebuild_store(self.conn, self.store)
        except Exception as exc:  # noqa: BLE001 - 启动不能因为对齐失败就起不来
            why = f"库里记的是「{recorded}」" if recorded else "之前没有登记过活动后端"
            self.notes.append(
                f"「{target}」尚未登记为活动后端（{why}），而它与归档的内容是否一致无法证明"
                f"——按归档重建也没成功（{_describe(exc)}）。本次不登记，检索与写入都会被拒绝；"
                "执行 `python -m rag4prefa rebuild` 修好后再重启"
            )
            return False
        why = f"库里记的是「{recorded}」" if recorded else "之前没有登记过活动后端"
        self.notes.append(
            f"已把归档的 {moved} 条向量重建进「{target}」并逐 id 校验（{why}）："
            "换后端时不重写内容就只能证明 id 相同，证明不了向量与过滤列是这一版"
        )
        return True

    def maybe_promote(self) -> bool:
        """启动时若因维度未知回退到 numpy，首次入库拿到维度后迁回 Milvus。

        避免"先写进 numpy、下次启动换成 Milvus 却查不到数据"的静默不一致。

        三件事必须一起成立，历史上各错一次：

        1. **只能由根服务执行**。请求级视图是 `replace()` 出来的临时对象，在它上面
           迁移会让"迁移成功、根服务还在用旧后端"，而且根服务的 `fell_back` 仍是
           True，下一个请求再迁一遍；
        2. **整个过程持有维护锁**，和入库、重建互斥——迁移要把归档搬进新后端，
           期间又有人写入，新后端拿到的就是过期快照；
        3. **异常一律截住**。上传接口是在 `ingest_batch` 之后调它的，一次没接住的
           异常会把已经 done 的任务改写成 failed，数据其实已经写好。
        """
        owner = self.owner
        if owner is not self:
            return owner.maybe_promote()
        if not self.fell_back or self.settings.vector_backend != "auto":
            return False
        try:
            with maintenance_lock(self.conn, purpose="迁移到 Milvus", timeout=0.0):
                return self._promote()
        except MaintenanceBusy as exc:
            # 有别的动作在改索引时不迁移：这次不是"失败"，是让路。下次入库还会再来。
            self.notes.append(f"暂不迁移到 Milvus：{exc}")
            return False

    def _promote(self) -> bool:
        dim = (self.embedder.dim if self.embedder else 0) or self.index_dim or 0
        if not dim:
            return False
        from .ingest import rebuild_store  # 局部导入，避免循环依赖

        try:
            # 已经在维护锁里（`_promote` 只从 `maybe_promote` 的锁内进入），所以这里
            # 的"打开索引"（含清理上次崩溃留下的暂存集合）是安全的。
            milvus = MilvusStore(
                self.settings.milvus_uri,
                self.settings.milvus_collection,
                dim,
                conn=self.conn,
            )
        except Exception as exc:  # noqa: BLE001 - 见方法说明：可选升级不能外溢
            # 以前只认 BackendUnavailable，但 pymilvus 在 Lite 文件被占用、版本不匹配时
            # 抛的是它自己的异常，那类失败同样只该降级、不该中断调用方。
            self.notes.append(f"仍无法启用 Milvus：{_describe(exc)}")
            return False

        try:
            moved = rebuild_store(self.conn, milvus)
        except Exception as exc:  # noqa: BLE001 - 同上
            # 迁移到 Milvus 只是"换个更快的后端"，是可选项。归档本身不合格、或 SDK 在
            # 建集合/写向量时报错，都留在原后端继续用，把原因记进 notes。
            # rebuild_store 只在校验通过后才动归档，所以这里不需要回收任何东西。
            self.notes.append(
                f"暂不迁移到 Milvus（继续用 {self.store.name} 检索）：{_describe(exc)}"
            )
            return False
        self.store = milvus
        self.fell_back = False
        # 活动后端已经由 `rebuild_store` 在"逐 id 校验通过 + 标记回 ready"的同一个
        # 事务里登记好了（`set_index_meta(backend=...)`）。这里再写一次属于重复，
        # 而且会在两个事务之间留下"索引已换、后端没换"的窗口。
        self.notes = [n for n in self.notes if "Milvus 不可用" not in n]
        self.notes.append(f"已把 {moved} 条归档向量迁移到 Milvus")
        return True

    def assert_query_ready(self) -> None:
        """检索准入的服务级判据：本进程到底确认过活动后端没有。

        两条证据，缺一不可：

        1. `backend_confirmed is not False`——启动时确实证明过（或库里早就是它）。
           `False` 意味着"试过、但没能证明"：抢不到维护锁、重建失败、读不到后端条数。
           这时按配置打开的那个后端**可能**是对的，但本进程拿不出证据，检索照旧会返回
           一份看起来正常的答案（2026-10-24 第十六轮复审）。
        2. 手里这个后端仍是库里登记的活动后端——长跑的服务不会因为别人换了后端就自动
           换过来，它只是继续查旧索引，一声不响。

        `None` 不拦：那表示"这个进程没跑确认流程"，出现在 CLI 与直接用 `Services` 的
        调用里。生产装配一定走 `build_services`，那里必跑 `settle_backend`。
        """
        if self.owner.backend_confirmed is False:
            raise IndexMismatch(
                "检索已拒绝：本进程启动时没能确认活动后端——它可能与归档的内容不一致"
                "（条数、模型、维度都能对得上，只有向量和过滤列不同）。"
                "原因见 `/api/health` 的 notes；用 `python -m rag4prefa rebuild` "
                "从归档恢复后重启服务"
            )
        ensure_active_backend(self.conn, self.store)

    def assert_write_ready(self) -> None:
        """普通写操作的准入：与检索同源，只是动作名不同。

        为什么入库/删除也要一道（2026-10-24 第十七轮复审）：`backend_confirmed` 是
        **本进程**的结论，`ensure_backend_current` 是**库里**的登记——两者缺一不可。

        - 只有前者：CLI 与上传入口没走这条路时，"试过但没证明"就只挡得住 API 查询；
        - 只有后者：库里登记着 milvus、本进程因为抢不到锁而没打开它时，登记名照样
          对得上，写入会落进一个 defer 状态的后端。

        诊断与修复动作（`doctor` / `verify` / `rebuild` / `reindex`）不调用它：它们
        本来就是用来把库修回可服务状态的，挡住它们等于让修复入口也进不去。
        """
        if self.owner.backend_confirmed is False:
            raise IndexMismatch(
                "写入已拒绝：本进程启动时没能确认活动后端——它可能与归档的内容不一致"
                "（条数、模型、维度都能对得上，只有向量和过滤列不同）。"
                "原因见 `/api/health` 的 notes；用 `python -m rag4prefa rebuild` "
                "从归档恢复后重启服务"
            )
        ensure_backend_current(self.conn, self.store)

    def _ocr_status(self) -> dict:
        """扫描件识别的当前状态：开没开、用哪个模型、能不能用、Key 从哪来。

        `ready` 是给前端用的判据：为假就表示"扫描件现在一定进不来"，界面要在
        用户拖文件之前就把配置入口摆出来，而不是让他等一个 20MB 的扫描件跑完
        再收到一句"不行"。

        **不报 Key 本身**，只报来源，与 `verify` 的 `source` 同一套词：
        `user` 是用户自己填的，`server` 是 `.env` 里那把，`missing` 是一把都没有。
        """
        eff = effective(self.settings)
        mode = self.settings.ocr_mode
        owned = bool(self.user_creds and self.user_creds.ocr_key)
        # "服务端那把 Key 存在"与"这次可以用"是两件事。不代付的部署里，`.env`
        # 那把只属于服务端主人：别的用户不填自己的 Key 就用不了扫描件。
        #
        # 为什么这里必须自己判一次，而不是靠 `settings_for_ocr` 清空：没带任何
        # `X-R4P-*` 头的根视图就是服务端主人自己，`for_credentials(None)` 返回的
        # settings 原样带着 `.env` 那把。只判 `eff.api_key` 会让页头显示
        # 「OCR 就绪」，而同一台机器上的访客上传扫描件其实会失败。
        payable = owned or self.settings.allow_server_key_fallback
        return {
            "mode": mode,
            "model": eff.model if mode != "off" else None,
            # 关掉与"开了但缺 Key"都会让扫描件进不来，但对用户是两件事：
            # 一个是没开，一个是开不了。前端据此说不同的话。
            "ready": bool(eff.ready and mode != "off" and payable),
            "key_source": (
                "user"
                if owned
                else ("server" if payable and eff.api_key else "missing")
            ),
        }

    def health(self) -> dict:
        # 后端条数是"索引 vs 归档"交叉校验的一半，读不到就如实说读不到，
        # 不要拿 None 当 0 —— 那会把"查不了"伪装成"索引是空的"。
        store_count: int | None
        store_error: str | None = None
        try:
            store_count = self.store.count()
        except Exception as exc:  # pragma: no cover - 后端不可用时才会走到
            store_count = None
            store_error = f"{type(exc).__name__}: {exc}"

        state = inspect(self.conn, store_count=store_count)
        issues = collect_issues(
            state, self.embedder.model if self.embedder else None, self.store.dim
        )
        if store_error:
            issues.append(f"读不到向量索引的实际条数，无法完成交叉校验：{store_error}")
        recorded = repo.get_vector_backend(self.conn)
        if recorded and recorded != self.store.name:
            # 条数、模型、维度全对也可能出现这个状态：换后端后本进程没重新装配。
            # 检索已被 `assert_query_ready` 挡住，这里要让运维先看见原因。
            issues.append(
                f"本进程持有的向量后端是「{self.store.name}」，库里登记的活动后端是"
                f"「{recorded}」——这个进程手里的索引停在切换之前，检索会被拒绝；重启服务"
            )
        if self.backend_confirmed is False:
            # 不健康，但要能启动：这时候 health 是唯一的诊断入口，它必须把"没证明"
            # 这件事说出来，而不是只留一条藏在下拉里的 note。
            issues.append(
                "本进程没能确认活动后端（启动时抢不到维护锁、重建失败或读不到后端条数）："
                "检索与普通写入都会拒绝，详见 notes"
            )
        return {
            # 谁在应答。停服命令（`rag4prefa stop`）靠它确认"端口上这个进程是不是
            # 本程序"——只按端口号杀进程，误杀别人家的服务只需要一次巧合。
            "app": "rag4prefa",
            # 索引与配置对不上时不能报"健康"：那种状态能跑，但检索结果不可信
            "ok": bool(self.embedder and self.llm) and not issues,
            "backend": {
                "name": self.store.name,
                # 库里登记的（别的进程也能看到）与本进程确认的，分开报：
                # 只报一个 name 会让人以为"打开的就是登记的那个"
                "recorded": recorded,
                # 身份 = 类型 + 存储位置：两个进程同名不同集合时，只有它能看出区别
                "identity": getattr(self.store, "identity", None),
                "recorded_identity": repo.get_vector_backend_id(self.conn),
                "confirmed": self.backend_confirmed,
                "vectors": store_count,
                "dim": self.store.dim,
                "fell_back": self.fell_back,
            },
            "index": {
                "model": state.model,
                "dim": state.dim,
                # 归档（事实）：片段的权威向量副本
                "vectors": state.count,
                "chunks": state.chunks,
                # 三个差额分开报，比"总数对不上"更能定位问题
                "missing_vectors": state.missing,
                "orphan_vectors": state.orphan,
                "unlabeled_vectors": state.unlabeled,
                "consistent": not issues,
                "issues": issues,
            },
            "credentials_missing": self.settings.missing_credentials(),
            # 扫描件能不能入库，取决于这一组开没开。不报出来的话，用户只会看到
            # "这个 pdf 几乎取不到文字"，然后自己去翻 .env。
            #
            # 档位必须一起报：同样写着 mode=auto，跑的可能是一页一分钱的付费模型，
            # 也可能是一分钱不花的免费模型。前端拿 `free` 决定要不要显示「零成本」。
            "ocr": self._ocr_status(),
            # 前端要拿它提示"你自己的 Key 必须属于同一家服务商"——模型由服务端统一决定
            "models": {
                "embedding_model": self.settings.embedding_model,
                "embedding_dim": self.settings.embedding_dim or self.store.dim,
                "llm_model": self.settings.llm_model,
            },
            # 让前端能证明"请求头确实被认到了"，而不是以为自己配了其实没用上
            "user": {
                "label": self.user_label or "服务端公共 Key",
                "source": "user" if self.user_label else "server",
            },
            # 没配重排服务时 reranker 是 Noop：勾了"启用重排"也什么都不做。
            # 前端必须知道这件事，否则那个开关就是一个安静的谎。
            "rerank_configured": not isinstance(self.reranker, NoopReranker),
            # 受控词表：开没开、几条、什么形态、坏了没有。
            # 只报 `enabled` 不报 `error`，"配了但没生效"就完全看不出来——
            # 表现是"界面一切正常，只有口语问句拿不到术语通道"。
            "term_map": {
                "enabled": self.term_map is not None,
                "path": self.settings.term_map,
                "entries": len(self.term_map) if self.term_map is not None else 0,
                "form": self.settings.term_map_form,
                # 没启用时报 `none`：这时重排读哪个串这件事根本不存在，
                # 报 `append` 会让人以为它在生效。
                "rerank": self.settings.term_rerank if self.term_map is not None else "none",
                "weight": self.settings.term_rewrite_weight,
                "error": self.term_map_error,
            },
            # 证据闸门：模式与两个门槛。
            #
            # 门槛报出来是因为"开着但门槛是 0"等于没开——`mode=on` 只说明代码会跑这段，
            # 不说明它真的会拒任何东西。排查"为什么库外问题照样回答了"时，第一眼看这里。
            "refusal": {
                "mode": self.settings.refuse_mode,
                "min_rerank": self.settings.refuse_min_rerank,
                "min_dense": self.settings.refuse_min_dense,
            },
            # 关掉回落时，不带头里的 Key 就调不了检索与问答。前端要据此把"用自己的 Key"
            # 从"可选"改成"必填"，而不是让用户点一次失败一次。
            "server_key_fallback": self.settings.allow_server_key_fallback,
            # 后端可能把恢复动作（如"已从上一版集合还原索引"）记在 notes 里；
            # 用 getattr 是为了让 health 在任何 VectorStore 实现上都不会因为缺属性而 500
            "notes": [*self.notes, *getattr(self.store, "notes", [])],
        }

    def close(self) -> None:
        # 视图（owns_conn=False）可能在一堆请求之间共享，关它等于把全局连接掐了
        if not self.owns_conn:
            return
        if self.credentials is not None:
            self.credentials.close()
        self.conn.close()


def open_store(
    settings: Settings, conn: sqlite3.Connection
) -> tuple[VectorStore, list[str], bool]:
    """按配置**打开**一个向量后端。

    三件事分开，是这一层最重要的一条（2026-10-24 第十五轮复审）：

    1. **打开**（这里）——只证明"这个后端能用"；
    2. **确认**（`Services.settle_backend`）——证明"它装的就是归档这一套"之后才登记；
    3. **换后端**（`rebuild_store`）——从归档整体重建到目标后端，校验通过才改登记。

    以前把 1 和 2 合成一步：只要 Milvus 能打开就写 `meta.vector_backend`。于是"遗留的
    旧集合"被当成当前版本——它与归档可以条数相同、chunk_id 全不同，日常计数校验一条也
    发现不了。

    **`auto` 尊重库里已经定下的选择**：库记着 numpy，说明有人显式切过去过（或上次是在
    没有 Milvus 的环境里确认的）。这时再"看到 Milvus 能开就切回去"，等于每次重启都
    推翻上一次的决定。只有显式 `R4P_VECTOR_BACKEND=milvus` 才是换回去。

    **`auto` 也不是"每个进程自己试"**：库里若已经确认在用 Milvus，这次就必须打开
    Milvus，打不开就报错。理由是回退不报错、后果却很难看——本进程退回 numpy 后，
    写入只更新"归档 + numpy 索引"，而在线服务的 Milvus 索引停在旧版。显式失败至少让人
    知道该修哪里。还没有确认过后端时（首次部署）才允许"试 Milvus，不行回退 numpy"。
    """
    notes: list[str] = []
    dim = settings.embedding_dim or int(get_meta(conn, "embedding_dim") or 0)
    recorded = get_vector_backend(conn)
    want_milvus = settings.vector_backend == "milvus" or (
        settings.vector_backend == "auto" and recorded != "numpy"
    )
    if want_milvus:
        try:
            # 传 conn：让"打开索引"（含崩溃恢复）走维护锁。构造函数本身只连客户端，
            # 有副作用的恢复动作由它自己按锁定边界执行——见 MilvusStore._open。
            store = MilvusStore(
                settings.milvus_uri, settings.milvus_collection, dim, conn=conn
            )
            # 这里**不登记**活动后端：`settle_backend()` 逐 id 核对过才会写。
            if recorded and recorded != "milvus":
                notes.append(
                    f"库里记录的活动后端是「{recorded}」，本次按配置打开了 Milvus；"
                    "在核对过它装的确实是归档这一套之前，不会把它登记为活动后端"
                )
            return store, notes, False
        except BackendUnavailable as exc:
            if settings.vector_backend == "milvus" or recorded == "milvus":
                why = (
                    "R4P_VECTOR_BACKEND=milvus"
                    if settings.vector_backend == "milvus"
                    else "库里记录的活动后端"
                )
                hint = (
                    f"{exc}。索引的活动后端是 Milvus（{why}），"
                    "不回退到 numpy：回退后本进程的写入只更新 numpy 索引，"
                    "在线服务查的 Milvus 会停在旧版，而两边条数一致时查不出来。"
                    "按上面的报错修好 Milvus；确认要换后端，就把它显式改成 numpy，"
                    "然后在启动后执行 `python -m rag4prefa rebuild` 重建一份 numpy 索引"
                    "（这条命令会把活动后端一并改成 numpy）。"
                )
                # "维度未知"只是 Milvus 打不开的众多原因之一。路径不合法、缺 pymilvus、
                # 中文路径，报的也是这一条——一律往"改维度"上引，会让人在不相关的
                # 配置项上反复试，越试越远。
                if "维度" in str(exc):
                    hint += "（报的是维度未知时：把 R4P_EMBEDDING_DIM 填成模型的真实维度。）"
                raise BackendUnavailable(hint) from exc
            notes.append(f"Milvus 不可用，已回退到 numpy 精确检索：{exc}")
            return NumpyStore(conn, dim), notes, True
    return NumpyStore(conn, dim), notes, False


def build_services(
    settings: Settings | None = None, *, recover_interrupted: bool | None = None
) -> Services:
    """装配服务。

    `recover_interrupted` 决定要不要把库里遗留的 `running` / `queued` 任务标成中断。
    默认值是"没别人在跑才标"：这个库现在同时被 uvicorn 和 CLI 使用，光看数据库
    分不出"上一个进程已经死了"和"另一个进程正在写"。以前每次装配都无条件清理，
    于是跑一次 `doctor` 就能把在线入库任务标成 failed，而数据其实还在写。

    判活的依据是 `concurrency.WORKER_LEASE`——服务进程活着就一直占着的文件锁，
    进程退出由内核释放，不需要心跳，也不会留下残留。
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)

    store, notes, fell_back = open_store(settings, conn)

    if recover_interrupted is None:
        recover_interrupted = not worker_alive(conn)
    if recover_interrupted:
        # 入库任务跑在进程内，上次进程死掉留下的 running/queued 永远不会前进
        interrupted = mark_interrupted_jobs(conn)
        if interrupted:
            notes.append(f"已把上次进程遗留的 {interrupted} 个任务标记为失败")
        # 同一类问题在文档维度也要收尾，否则它会永远显示"正在入库"
        dangling = mark_interrupted_documents(conn)
        if dangling:
            notes.append(
                f"已把上次进程中断时处于入库中的 {dangling} 个文档标记为失败，可重新导入"
            )
    else:
        # 说清"为什么这次不动它们"，否则运维会以为清理逻辑失效了
        notes.append(
            "检测到另一个服务进程正在使用同一个数据库，已跳过对未完成任务的中断清理"
            "（那些任务可能正在别的进程里跑）"
        )

    embedder: Embedder | None
    try:
        embedder = Embedder(settings, conn)
    except EmbeddingError as exc:
        embedder = None
        notes.append(str(exc))

    llm: LLM | None
    try:
        llm = LLM(settings)
    except LLMError as exc:
        llm = None
        notes.append(str(exc))

    reranker: Reranker = RemoteReranker(settings) if settings.rerank_url else NoopReranker()

    # 词表读一次就够。坏词表**不让服务起不来**（它是可选增强项，一个写错的 JSON
    # 不该把检索整个堵死），但也绝不静默：原因进 notes 与 `/api/health` 的
    # `term_map.error`，`term_map` 留 None 让检索退回"没有这一层"。
    # 两件事必须同时成立——只降级不报错，等于让人以为词表在生效。
    term_map: TermMap | None = None
    term_map_error: str | None = None
    try:
        term_map = load_term_map(settings.term_map or None)
    except (OSError, ValueError) as exc:
        term_map_error = f"{type(exc).__name__}: {exc}"
        notes.append(
            f"受控词表未能加载（{term_map_error}），本次按「不启用词表」运行："
            "检索仍然可用，但口语问句拿不到术语通道"
        )

    services = Services(
        settings=settings,
        conn=conn,
        store=store,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
        notes=notes,
        fell_back=fell_back,
        term_map=term_map,
        term_map_error=term_map_error,
    )
    # 打开 ≠ 确认：能在集合打开只说明它存在，说明不了它装的是归档这一套。
    # 这一步在需要时会在维护锁里从归档重建，然后才把后端登记进 meta。
    services.settle_backend()
    return services


def default_upload_dir(settings: Settings) -> Path:
    return settings.upload_dir
