import gradio as gr
from glob import glob
from random import shuffle
from collections import defaultdict
from datetime import datetime

# audio_list_f = glob("test/f_*.wav")
# audio_list_f.sort(key=lambda x: x[::-1])
audio_list_m = glob("mos_wav/*.wav")
audio_list_m.sort()
data = []
# for i in range(0, len(audio_list_f), 2):
#     d = [audio_list_f[i], audio_list_f[i+1]]
#     shuffle(d)
#     data.append(d)
for i in range(0, len(audio_list_m), 2):
    d = [audio_list_m[i], audio_list_m[i+1]]
    shuffle(d)
    data.append(d)

def who_win(sample, pref):
    if pref == 0: idx = 0
    elif pref == 2: idx = 1
    else: return "draw"

    if "genb" in sample[idx]: return "uni_pos"
    if "gen" in sample[idx]: return "bi_pos"

def create_interface(audio_list):
    def report(*prefs):
        result = defaultdict(int)
        for p, d in zip(prefs, data):
            result[who_win(d, p)] += 1
        filename = "mos_result_unid_vs_bid/" + format(datetime.now()).replace(" ", "_").replace(".", "_") + ".txt"
        with open(filename, "w") as fw:
            fw.write(f"result: {dict(result)}")
        return f"result: {dict(result)} / 감사합니다!"
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
        submit_button.click(report, inputs=prefs, outputs=output)

    demo.launch(server_name="0.0.0.0", share=True)

create_interface(data)