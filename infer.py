from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
import argparse
import flash_attn
import torch
import time
#todo:他是把音频先过一遍大模型再当成token过一遍大模型，看一看是不是变成文字，看看有没有硬性的时间指标
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    args = parser.parse_args()

    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=True,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    # messages = [
    #     {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "my_test/hit_q.wav",
    #     },
    # ]
    # start = time.time()
    # wav, text = model.generate(messages, **sampling_params, output_type="text")
    # end = time.time()
    # print("-"*20)
    # print("audio2text:",end-start)
    # print("-"*20)
    # print(">>> output text: ", text)

    output_dir = "my_test/output"
    # os.makedirs(output_dir, exist_ok=True)
    # audio2audio
    messages = [
        {
            "role": "user",
            "message_type": "audio",
            "content": "my_test/hit_q.wav",
        }
    ]
    start = time.time()
    wav, text = model.generate(messages, **sampling_params, output_type="both")
    end = time.time()
    print("-"*20)
    print("audio2audio_q:",end-start)
    print("-"*20)
    print(">>> output text: ", text)
    sf.write(
        os.path.join(output_dir, "hit_q_output_temp.wav"),
        wav.detach().cpu().view(-1).numpy(),
        24000,
    )
    
    # audio2audio
    i=10
    while(i<150):
        messages = [
            {
                "role": "user",
                "message_type": "audio",
                "content": "my_test/hit_q.wav",
            }
        ]
        start = time.time()
        wav, text = model.generate(messages, **sampling_params, output_type="both",number_of_tokens_to_a_wav=i)
        end = time.time()
        print("-"*20)
        print("audio2audio_q:",end-start)
        print("-"*20)
        print(">>> output text: ", text)
        sf.write(
            os.path.join(output_dir, "hit_q_output_short.wav"),
            wav.detach().cpu().view(-1).numpy(),
            24000,
        )
        i+=10
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    # # audio2audio
    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "my_test/hitc_q.wav",
    #     }
    # ]
    # start = time.time()
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # end = time.time()
    # print("-"*20)
    # print("audio2audio_cq:",end-start)
    # print("-"*20)
    # print(">>> output text: ", text)
    # sf.write(
    #     os.path.join(output_dir, "hitc_q_output.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )

    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "my_test/hits_q.wav",
    #     }
    # ]
    # start = time.time()
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # end = time.time()
    # print("-"*20)
    # print("audio2audio_sq:",end-start)
    # print("-"*20)
    # print(">>> output text: ", text)
    # sf.write(
    #     os.path.join(output_dir, "hits_q_output.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # audio2audio multiturn
    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "my_test/hit_q.wav",
    #     },
    #     {
    #         "role": "assistant",
    #         "message_type": "audio-text",
    #         "content": [os.path.join(output_dir, "hit_q_output.wav"), "哈尔滨工业大学，简称哈工大，是中国顶尖的工科学校之一，位于黑龙江省哈尔滨市。这学校历史挺悠久的，1920年就有了，是中国最早的国立大学之一。哈工大以工程学科见长，特别是在航天、机械、材料这些领域特别牛。学校里有很多厉害的教授和科研项目，培养出了不少杰出的工程师和科学家。校园环境也很美，挺适合学习的。"]
    #     },
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "my_test/hitc_q.wav",
    #     }
    # ]
    # start = time.time()
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # end = time.time()
    # print("-"*20)
    # print("multiturn:",end-start)
    # print("-"*20)
    # sf.write(
    #     os.path.join(output_dir, "hitc_multiturn_output.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # print(">>> output text: ", text)
"""
    


    messages = [
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/multiturn/case2/multiturn_q1.wav",
        },
        {
            "role": "assistant",
            "message_type": "audio-text",
            "content": ["test_audios/multiturn/case2/multiturn_a1.wav", "当然可以，这很简单。一二三四五六七八九十。"]
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": "test_audios/multiturn/case2/multiturn_q2.wav",
        }
    ]
    wav, text = model.generate(messages, **sampling_params, output_type="both")
    sf.write(
        os.path.join(output_dir, "case_2_multiturn_a2.wav"),
        wav.detach().cpu().view(-1).numpy(),
        24000,
    )
    print(">>> output text: ", text)

    """
