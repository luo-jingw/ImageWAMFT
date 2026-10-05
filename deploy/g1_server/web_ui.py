"""Gradio page of the G1 policy server: model and checkpoint selection, inference settings, the last request and the
request log. Every action goes through `PolicyManager`; the page holds no policy state of its own."""

import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from g1_server.inference_settings import TRAINED_ACTION_HORIZON, InferenceSettings  # noqa: E402
from g1_server.model_registry import MODEL_SPECS, available_steps  # noqa: E402
from g1_server.policy_manager import PolicyManager, RequestRecord  # noqa: E402
from g1_server.protocol import CAMERA_KEYS, ObservationRequest  # noqa: E402

plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 9})
LABEL_TO_NAME = {spec.label: spec.name for spec in MODEL_SPECS}
SEED_FIXED, SEED_RANDOM = "固定种子", "每次随机"
PROMPT_CLIENT, PROMPT_FIXED = "使用机器人端发来的指令", "使用下面选定的指令"
LOG_HEADERS = ["#", "时间", "客户端", "模型", "step", "去噪步数", "种子", "指令来源", "总耗时 ms", "模型 ms", "错误"]
ARM_NAMES = ("pitch", "roll", "yaw", "elbow", "wrist roll", "wrist pitch", "wrist yaw")


class WebUI:
    def __init__(self, manager: PolicyManager, websocket_url: str) -> None:
        self.manager = manager
        self.websocket_url = websocket_url

    # ---- model selection -------------------------------------------------------------------------------------
    def step_choices(self, label: str) -> gr.Dropdown:
        steps = [str(step) for step in available_steps(next(s for s in MODEL_SPECS if s.label == label))]
        return gr.Dropdown(choices=steps, value=steps[-1] if steps else None)

    def load(self, label: str, step: str) -> str:
        name = LABEL_TO_NAME[label]
        resident = {policy.name: policy.step for policy in self.manager.status().resident}
        if resident.get(name) != int(step):
            self.manager.load(name, int(step))
        self.manager.activate(name)
        return self.status_text()

    def unload(self, label: str) -> str:
        self.manager.unload(LABEL_TO_NAME[label])
        return self.status_text()

    def status_text(self) -> str:
        status = self.manager.status()
        labels = {spec.name: spec.label for spec in MODEL_SPECS}
        active = next((p for p in status.resident if p.name == status.active), None)
        resident = "；".join(f"{p.label} step {p.step}（加载 {p.load_seconds:.0f} s）" for p in status.resident) or "无"
        lines = [
            f"**机器人端连接地址**：`{self.websocket_url}`",
            f"**当前服务的模型**：{labels[active.name]} step {active.step}" if active else "**当前服务的模型**：无（请先加载）",
            f"**已加载（常驻显存）**：{resident}",
            f"**GPU 显存**：已分配 {status.gpu_memory_allocated_gib:.1f} GiB，已保留 {status.gpu_memory_reserved_gib:.1f} GiB",
            f"**请求**：推理成功 {status.requests_served}，失败 {status.requests_failed}；相机预览 {status.previews}",
        ]
        if status.busy:
            lines.append(f"**正在进行**：{status.busy}")
        return "\n\n".join(lines)

    # ---- inference settings ----------------------------------------------------------------------------------
    def apply_settings(self, steps: float, shift_default: bool, shift: float, seed_mode: str, seed: float,
                       horizon: float, prompt_mode: str, task: str) -> str:
        settings = InferenceSettings(num_inference_steps=int(steps), sigma_shift=None if shift_default else float(shift),
                                     seed=int(seed) if seed_mode == SEED_FIXED else None, action_horizon=int(horizon),
                                     prompt_override=task.strip() if prompt_mode == PROMPT_FIXED and task else None)
        self.manager.update_settings(settings)
        return self.settings_text()

    def settings_text(self) -> str:
        s = self.manager.settings()
        shift = "run 默认（infer_shift）" if s.sigma_shift is None else f"{s.sigma_shift:g}"
        seed = "每次随机" if s.seed is None else str(s.seed)
        prompt = "机器人端发来的指令" if s.prompt_override is None else f"固定：{s.prompt_override}"
        horizon = f"{s.action_horizon}" + ("" if s.action_horizon == TRAINED_ACTION_HORIZON else "（训练时为 16）")
        return (f"当前生效：去噪步数 {s.num_inference_steps}；sigma shift {shift}；种子 {seed}；动作块长度 {horizon}；"
                f"指令：{prompt}")

    # ---- last request ----------------------------------------------------------------------------------------
    def refresh(self, shown_index: int) -> tuple:
        records = self.manager.recent(50)
        status = self.status_text()
        if not records or records[-1].index == shown_index:
            return status, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), shown_index
        last = records[-1]
        views = [(view, key) for view, key in zip(last.views, CAMERA_KEYS)]
        if last.preview_image is not None:  # camera preview: no action chunk, keep the last plot and log
            return status, views, last.preview_image, gr.skip(), self.record_text(last), gr.skip(), last.index
        image = last.output.model_image if last.output else None
        plot = self.action_figure(last) if last.output else None
        inference = [r for r in records if r.preview_image is None][-20:]
        return status, views, image, plot, self.record_text(last), self.log_rows(inference), last.index

    def record_text(self, record: RequestRecord) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(record.wall_time))
        if record.preview_image is not None:
            return f"**相机预览**（不推理、机器人不动）{when} 来自 `{record.client}`"
        if record.output is None:
            return f"**#{record.index}** {when} 来自 `{record.client}`：失败 — {record.error}"
        out = record.output
        t = out.timing
        return (f"**#{record.index}** {when} 来自 `{record.client}` ｜ {out.model} step {out.step} ｜ 种子 {out.seed}，"
                f"去噪 {out.num_inference_steps} 步 ｜ 耗时：预处理 {t.preprocess_ms:.0f} ms，文本 {t.text_ms:.0f} ms，"
                f"模型 {t.model_ms:.0f} ms，合计 {t.total_ms:.0f} ms\n\n**指令**（{out.prompt_source}）：{out.task}")

    @staticmethod
    def action_figure(record: RequestRecord) -> Figure:
        actions, state = record.output.actions, record.state
        steps = np.arange(actions.shape[0])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.7]})
        colors = plt.get_cmap("tab10").colors
        for ax, offset, title in ((axes[0], 0, "左臂关节 (rad)"), (axes[1], 7, "右臂关节 (rad)")):
            for j in range(7):
                ax.plot(steps, actions[:, offset + j], color=colors[j], linewidth=1.6, label=ARM_NAMES[j])
                ax.plot([-1], [state[offset + j]], "o", color=colors[j], markersize=4)
            ax.set_title(title)
        for j, name in ((14, "左夹爪"), (15, "右夹爪")):
            axes[2].plot(steps, actions[:, j], color=colors[j - 14], linewidth=1.6, label=name)
            axes[2].plot([-1], [state[j]], "o", color=colors[j - 14], markersize=4)
        axes[2].set_title("夹爪")
        axes[0].legend(fontsize=7, ncol=2, loc="best")
        axes[2].legend(fontsize=7)
        for ax in axes:
            ax.set_xlabel("动作步（-1 处的点 = 当前状态）")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.close(fig)
        return fig

    @staticmethod
    def log_rows(records: list[RequestRecord]) -> list[list]:
        rows = []
        for r in reversed(records):
            out = r.output
            rows.append([r.index, time.strftime("%H:%M:%S", time.localtime(r.wall_time)), r.client,
                         out.model if out else r.model, r.step, r.settings.num_inference_steps,
                         out.seed if out else "", out.prompt_source if out else "",
                         round(out.timing.total_ms) if out else "", round(out.timing.model_ms) if out else "",
                         r.error or ""])
        return rows

    def predict_image(self) -> tuple:
        records = [r for r in self.manager.recent(50) if r.output is not None]
        if not records:
            raise gr.Error("还没有成功的推理请求；预测图基于最近一次推理请求的观测")
        last = records[-1]
        request = ObservationRequest(views=last.views, state=last.state, prompt=last.output.task)
        return last.output.model_image, self.manager.predict_image(request)

    # ---- layout ----------------------------------------------------------------------------------------------
    def build(self) -> gr.Blocks:
        settings = self.manager.settings()
        status = self.manager.status()
        active_label = next((p.label for p in status.resident if p.name == status.active), MODEL_SPECS[0].label)
        steps = [str(step) for step in available_steps(next(s for s in MODEL_SPECS if s.label == active_label))]
        active_step = next((str(p.step) for p in status.resident if p.name == status.active), steps[-1])
        with gr.Blocks(title="ImageWAM G1 推理服务", analytics_enabled=False) as page:
            gr.Markdown("## ImageWAM G1 推理服务")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型")
                    model = gr.Dropdown([s.label for s in MODEL_SPECS], value=active_label, label="模型")
                    step = gr.Dropdown(steps, value=active_step, label="checkpoint step", allow_custom_value=False)
                    with gr.Row():
                        load = gr.Button("加载并切换到此模型", variant="primary")
                        unload = gr.Button("卸载此模型")
                    status_md = gr.Markdown(self.status_text())
                with gr.Column(scale=1):
                    gr.Markdown("### 推理设置（对之后的请求生效）")
                    num_steps = gr.Slider(1, 50, value=settings.num_inference_steps, step=1, label="去噪步数（训练验证用 10）")
                    with gr.Row():
                        shift_default = gr.Checkbox(value=settings.sigma_shift is None, label="sigma shift 用 run 默认值")
                        shift = gr.Number(value=settings.sigma_shift or 5.0, label="sigma shift")
                    with gr.Row():
                        seed_mode = gr.Radio([SEED_FIXED, SEED_RANDOM], value=SEED_FIXED if settings.seed is not None
                                             else SEED_RANDOM, label="初始噪声")
                        seed = gr.Number(value=settings.seed if settings.seed is not None else 42, precision=0,
                                         label="种子")
                    horizon = gr.Slider(1, 64, value=settings.action_horizon, step=1, label="动作块长度（训练时 16）")
                    prompt_mode = gr.Radio([PROMPT_CLIENT, PROMPT_FIXED], value=PROMPT_CLIENT, label="指令来源")
                    task = gr.Dropdown(self.manager.tasks(), value=None, allow_custom_value=True,
                                       label="指令（训练集的 4 条任务，也可以输入新指令）")
                    apply = gr.Button("应用设置", variant="primary")
                    settings_md = gr.Markdown(self.settings_text())
            gr.Markdown("### 最近一次请求（推理请求或相机预览，每 0.5 秒刷新）")
            record_md = gr.Markdown("还没有请求")
            with gr.Row():
                views = gr.Gallery(label="机器人端发来的三路相机", columns=3, height=260)
                model_image = gr.Image(label="模型输入（288×256）", type="numpy", height=260)
            plot = gr.Plot(label="动作块")
            with gr.Row():
                predict = gr.Button("用最近一次观测预测 16 步后的画面")
            with gr.Row():
                predict_input = gr.Image(label="模型输入", type="numpy", height=300)
                predict_output = gr.Image(label="模型预测的 16 步后画面", type="numpy", height=300)
            log = gr.Dataframe(headers=LOG_HEADERS, label="请求记录（最近 20 条）", interactive=False)
            shown = gr.State(-1)

            model.change(self.step_choices, model, step)
            load.click(self.load, [model, step], status_md)
            unload.click(self.unload, model, status_md)
            apply.click(self.apply_settings, [num_steps, shift_default, shift, seed_mode, seed, horizon, prompt_mode,
                                              task], settings_md)
            predict.click(self.predict_image, None, [predict_input, predict_output])
            gr.Timer(0.5).tick(self.refresh, shown, [status_md, views, model_image, plot, record_md, log, shown])
        return page


def build_web_ui(manager: PolicyManager, websocket_url: str) -> gr.Blocks:
    return WebUI(manager, websocket_url).build()
