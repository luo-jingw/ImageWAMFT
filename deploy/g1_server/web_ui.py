"""Gradio page of the G1 policy server: safety controls (output mode, action safety check, WebSocket port and client
connections), model and checkpoint selection, inference settings, the last request and the request log. Every action
goes through `PolicyManager` or `PolicyWebSocketServer`; the page holds no policy state of its own."""

import time
from dataclasses import replace

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
from g1_server.serving_controls import OutputMode  # noqa: E402
from g1_server.websocket_server import PolicyWebSocketServer  # noqa: E402

plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "font.size": 9})
LABEL_TO_NAME = {spec.label: spec.name for spec in MODEL_SPECS}
SEED_FIXED, SEED_RANDOM = "固定种子", "每次随机"
PROMPT_CLIENT, PROMPT_FIXED = "使用机器人端发来的指令", "使用下面选定的指令"
MODE_LABELS = {OutputMode.INFERENCE_ONLY: "仅推理（机器人保持不动）", OutputMode.EXECUTE: "执行（下发模型动作）",
               OutputMode.STOP: "停止（拒绝请求，客户端退出）"}
LABEL_TO_MODE = {label: mode for mode, label in MODE_LABELS.items()}
MODE_BANNER = {
    OutputMode.INFERENCE_ONLY: ("#9a6a00", "#fff4d6", "仅推理：照常推理并在本页显示，下发给机器人的是“保持当前姿态”，机器人不动"),
    OutputMode.EXECUTE: ("#0b6b3a", "#dcf5e7", "执行：通过安全检查的模型动作会下发给机器人，机器人会动"),
    OutputMode.STOP: ("#a3200f", "#fde2dd", "停止：不推理，拒绝所有观测请求，机器人端会报错退出并交还手臂控制权"),
}
SOURCE_TEXT = {"model": "模型动作", "hold": "保持不动"}
LOG_HEADERS = ["#", "时间", "客户端", "模型", "step", "模式", "下发", "安全检查", "第一步跳变 rad", "去噪步数", "种子",
               "总耗时 ms", "错误"]
CLIENT_HEADERS = ["客户端", "连接时间", "请求数"]
ARM_NAMES = ("pitch", "roll", "yaw", "elbow", "wrist roll", "wrist pitch", "wrist yaw")


class WebUI:
    def __init__(self, manager: PolicyManager, server: PolicyWebSocketServer) -> None:
        self.manager = manager
        self.server = server

    # ---- safety controls -------------------------------------------------------------------------------------
    def banner(self) -> str:
        controls = self.manager.controls()
        color, background, text = MODE_BANNER[controls.mode]
        check = ("开启" if controls.safety_check else "<b>关闭</b>")
        return (f'<div style="border:2px solid {color};background:{background};color:{color};padding:10px 14px;'
                f'border-radius:6px;font-size:16px"><b>当前输出模式 — {text}</b><br>'
                f'<span style="font-size:13px">动作安全检查：{check}（范围余量 {controls.range_margin:.0%}，'
                f'第一步最大跳变 {controls.max_first_step_jump:.2f} rad）｜机器人端连接地址：'
                f'<code>{self.server.url()}</code></span></div>')

    def apply_mode(self, label: str, confirmed: bool) -> tuple:
        mode = LABEL_TO_MODE[label]
        if mode is OutputMode.EXECUTE and not confirmed:
            raise gr.Error("切换到“执行”前，请先勾选确认机器人周围安全")
        self.manager.update_controls(replace(self.manager.controls(), mode=mode))
        return self.banner(), gr.update(value=False)

    def stop_now(self) -> tuple:
        self.manager.update_controls(replace(self.manager.controls(), mode=OutputMode.STOP))
        return self.banner(), gr.update(value=MODE_LABELS[OutputMode.STOP]), gr.update(value=False)

    def apply_safety(self, enabled: bool, margin_percent: float, max_jump: float) -> str:
        controls = replace(self.manager.controls(), safety_check=bool(enabled), range_margin=float(margin_percent) / 100,
                           max_first_step_jump=float(max_jump))
        try:
            self.manager.update_controls(controls)
        except ValueError as error:
            raise gr.Error(str(error))
        return self.banner()

    def change_port(self, port: float) -> str:
        try:
            self.server.change_port(int(port))
        except (ValueError, OSError) as error:
            raise gr.Error(f"端口没有修改，仍在 {self.server.port}：{error}")
        return self.banner()

    def disconnect_all(self) -> str:
        closed = self.server.disconnect_all()
        gr.Info(f"已断开 {closed} 个连接")
        return self.banner()

    def client_rows(self) -> list[list]:
        return [[c.address, time.strftime("%H:%M:%S", time.localtime(c.connected_at)), c.requests]
                for c in self.server.connections()]

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
            f"**当前服务的模型**：{labels[active.name]} step {active.step}" if active else "**当前服务的模型**：无（请先加载）",
            f"**已加载（常驻显存）**：{resident}",
            f"**GPU 显存**：已分配 {status.gpu_memory_allocated_gib:.1f} GiB，已保留 {status.gpu_memory_reserved_gib:.1f} GiB",
            f"**请求**：推理成功 {status.requests_served}（其中下发“保持不动” {status.holds_sent}，"
            f"安全检查未通过 {status.safety_flags}），失败 {status.requests_failed}，停止模式拒绝 {status.stopped}；"
            f"相机预览 {status.previews}",
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
        fixed = (self.status_text(), self.banner(), self.client_rows())
        if not records or records[-1].index == shown_index:
            return fixed + (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), shown_index)
        last = records[-1]
        views = [(view, key) for view, key in zip(last.views, CAMERA_KEYS)]
        if last.preview_image is not None:  # camera preview: no action chunk, keep the last plot and log
            return fixed + (views, last.preview_image, gr.skip(), self.record_text(last), gr.skip(), last.index)
        image = last.output.model_image if last.output else None
        plot = self.action_figure(last) if last.output else None
        inference = [r for r in records if r.preview_image is None][-20:]
        return fixed + (views, image, plot, self.record_text(last), self.log_rows(inference), last.index)

    @staticmethod
    def delivery_text(record: RequestRecord) -> str:
        if record.mode is OutputMode.STOP:
            return "已拒绝（停止模式）"
        if record.actions_source is None:
            return ""
        if record.actions_source == "model":
            return "模型动作"
        if record.mode is OutputMode.INFERENCE_ONLY:
            return "保持不动（仅推理）"
        return "保持不动（安全检查未通过）"

    def record_text(self, record: RequestRecord) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(record.wall_time))
        if record.preview_image is not None:
            return f"**相机预览**（不推理、机器人不动）{when} 来自 `{record.client}`"
        if record.output is None:
            return f"**#{record.index}** {when} 来自 `{record.client}`：{self.delivery_text(record) or '失败'} — {record.error}"
        out = record.output
        t = out.timing
        verdict = record.verdict
        safety = ("通过" if verdict.ok else '<span style="color:#a3200f"><b>未通过：' + "；".join(verdict.reasons)
                  + "</b></span>")
        return (f"**#{record.index}** {when} 来自 `{record.client}` ｜ {out.model} step {out.step} ｜ 种子 {out.seed}，"
                f"去噪 {out.num_inference_steps} 步 ｜ 耗时：预处理 {t.preprocess_ms:.0f} ms，文本 {t.text_ms:.0f} ms，"
                f"模型 {t.model_ms:.0f} ms，合计 {t.total_ms:.0f} ms\n\n"
                f"**下发给机器人**：{self.delivery_text(record)} ｜ **安全检查**：{safety}（第一步手臂最大跳变 "
                f"{verdict.max_first_step_jump:.3f} rad）\n\n**指令**（{out.prompt_source}）：{out.task}")

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
        sent = "（已下发）" if record.actions_source == "model" else "（未下发，机器人保持不动）"
        fig.suptitle(f"模型预测的动作块{sent}", x=0.01, ha="left", fontsize=10)
        for ax in axes:
            ax.set_xlabel("动作步（-1 处的点 = 当前状态）")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def log_rows(self, records: list[RequestRecord]) -> list[list]:
        rows = []
        for r in reversed(records):
            out, verdict = r.output, r.verdict
            rows.append([r.index, time.strftime("%H:%M:%S", time.localtime(r.wall_time)), r.client,
                         out.model if out else r.model, r.step, r.mode.value if r.mode else "",
                         self.delivery_text(r), ("通过" if verdict.ok else "未通过") if verdict else "",
                         round(verdict.max_first_step_jump, 3) if verdict else "", r.settings.num_inference_steps,
                         out.seed if out else "", round(out.timing.total_ms) if out else "", r.error or ""])
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
        controls = self.manager.controls()
        status = self.manager.status()
        active_label = next((p.label for p in status.resident if p.name == status.active), MODEL_SPECS[0].label)
        steps = [str(step) for step in available_steps(next(s for s in MODEL_SPECS if s.label == active_label))]
        active_step = next((str(p.step) for p in status.resident if p.name == status.active), steps[-1])
        with gr.Blocks(title="ImageWAM G1 推理服务", analytics_enabled=False) as page:
            gr.Markdown("## ImageWAM G1 推理服务")
            banner = gr.HTML(self.banner())
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输出模式（对下一个请求生效）")
                    mode = gr.Radio(list(MODE_LABELS.values()), value=MODE_LABELS[controls.mode], label="输出模式")
                    confirm = gr.Checkbox(value=False, label="我确认机器人周围安全，允许执行模型动作（切到“执行”时必须勾选）")
                    with gr.Row():
                        apply_mode = gr.Button("应用输出模式", variant="primary")
                        stop = gr.Button("立即停止", variant="stop")
                with gr.Column(scale=1):
                    gr.Markdown("### 动作安全检查（未通过时下发“保持不动”）")
                    safety_on = gr.Checkbox(value=controls.safety_check, label="启用动作安全检查")
                    with gr.Row():
                        margin = gr.Number(value=round(controls.range_margin * 100), precision=0,
                                           label="训练动作范围的余量（%）")
                        max_jump = gr.Number(value=controls.max_first_step_jump, label="第一步手臂最大跳变（rad）")
                    apply_safety = gr.Button("应用安全检查设置")
                with gr.Column(scale=1):
                    gr.Markdown("### 连接")
                    with gr.Row():
                        port = gr.Number(value=self.server.port, precision=0, label="WebSocket 端口（1024–65535）")
                        change_port = gr.Button("修改端口")
                    clients = gr.Dataframe(headers=CLIENT_HEADERS, value=self.client_rows(), label="当前连接的客户端",
                                           interactive=False)
                    disconnect = gr.Button("断开所有连接", variant="stop")
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

            apply_mode.click(self.apply_mode, [mode, confirm], [banner, confirm])
            stop.click(self.stop_now, None, [banner, mode, confirm])
            apply_safety.click(self.apply_safety, [safety_on, margin, max_jump], banner)
            change_port.click(self.change_port, port, banner)
            disconnect.click(self.disconnect_all, None, banner)
            model.change(self.step_choices, model, step)
            load.click(self.load, [model, step], status_md)
            unload.click(self.unload, model, status_md)
            apply.click(self.apply_settings, [num_steps, shift_default, shift, seed_mode, seed, horizon, prompt_mode,
                                              task], settings_md)
            predict.click(self.predict_image, None, [predict_input, predict_output])
            gr.Timer(0.5).tick(self.refresh, shown, [status_md, banner, clients, views, model_image, plot, record_md,
                                                     log, shown])
        return page


def build_web_ui(manager: PolicyManager, server: PolicyWebSocketServer) -> gr.Blocks:
    return WebUI(manager, server).build()
