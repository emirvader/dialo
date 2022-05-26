import audioop
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

engine = pyttsx3.init()

for voice in engine.getProperty("voices"):
    print(voice)

voices = engine.getProperty("voices")

engine.setProperty("voice", voices[1].id)

def main(Audio1):

    r = sr.Recognizer()

    step = 0

    engine.say(Audio1)
    engine.runAndWait()

    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)

        print("Please say something")

        audio = r.listen(source)

        print("Recognizing Now .... ")


        # recognize speech using google

        try:

            mic_in = str(r.recognize_google(audio))
            
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = tokenizer.encode(mic_in + tokenizer.eos_token, return_tensors='pt')

            # append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

            # generated a response while limiting the total chat history to 1000 tokens, 
            chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

            # pretty print last ouput tokens from bot
            output = ("{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

            main(output)
            


        except Exception as e:
            # print("Error :  " + str(e))
            pass
        return main




        # write audio
        # with open("recorded.wav", "wb") as f:
        #     f.write(audio.get_wav_data())


if __name__ == "__main__":
    main(audioop)