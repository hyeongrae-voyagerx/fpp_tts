import gradio as gr
from glob import glob
from random import shuffle
from collections import defaultdict

audio_list_f = glob("test/f_*.wav")
audio_list_f.sort(key=lambda x: x[::-1])
audio_list_m = glob("test/m_*.wav")
audio_list_m.sort(key=lambda x: x[::-1])
data = []
for i in range(0, len(audio_list_f), 2):
    d = [audio_list_f[i], audio_list_f[i+1]]
    shuffle(d)
    data.append(d)
for i in range(0, len(audio_list_m), 2):
    d = [audio_list_m[i], audio_list_m[i+1]]
    shuffle(d)
    data.append(d)

def who_win(sample, pref):
    if pref == 0: idx = 0
    elif pref == 2: idx = 1
    else: return "draw"

    if "baseline" in sample[idx]: return "baseline"
    if "gen" in sample[idx]: return "gen"

def create_interface(audio_list):
    def play_audio(*prefs):
        result = defaultdict(int)
        for p, d in zip(prefs, data):
            result[who_win(d, p)] += 1
        return f"result: {dict(result)}"
    prefs = []
    with gr.Blocks() as demo:
        for audio_pair in audio_list:
            with gr.Row():
                with gr.Column():
                    left_audio = gr.Audio(audio_pair[0], label="A", autoplay=False)

                with gr.Column():
                    preference = gr.Radio(
                        ["A is better", "Same", "B is better"],
                        label="선호도",
                        type="index",
                    )
                    prefs.append(preference)

                with gr.Column():
                    right_audio = gr.Audio(audio_pair[1], label="B", autoplay=False)

        # 제출 버튼 클릭 시 선호도를 처리
        output = gr.Textbox(label="결과")
        submit_button = gr.Button("제출")
        submit_button.click(play_audio, inputs=prefs, outputs=output)

    demo.launch()

create_interface(data)