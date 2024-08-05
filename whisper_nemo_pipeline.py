import os
import torch
from library import (
    getAllFiles,
    transcribe_batched,
    transcribe,
    wav2vec2_langs,
    filter_missing_timestamps,
    create_config,
    get_words_speaker_mapping,
    punct_model_langs,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    write_srt,
    updateStatus,
)
import whisperx
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import logging
import re


# PIPELINE STARTS HERE


def diarize(file):
    (id, audio_path, filename, showname, episode, title, duration, status) = file
    print(f"processing {filename}...")
    vocal_target = audio_path

    whisper_model_name = "large-v3"
    suppress_numerals = True
    batch_size = 8
    language = "en"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcribe using Whisper and reallign timestamps using Wav2Vec2

    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type = "int8"

    if batch_size != 0:
        whisper_results, language = transcribe_batched(
            vocal_target,
            language,
            batch_size,
            whisper_model_name,
            compute_type,
            suppress_numerals,
            device,
        )
    else:
        whisper_results, language = transcribe(
            vocal_target,
            language,
            whisper_model_name,
            compute_type,
            suppress_numerals,
            device,
        )

        # Align transcription with original audio using Wav2Vec2

    if language in wav2vec2_langs:
        device = "cuda"
        alignment_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, device
        )
        word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()
    else:
        assert (
            batch_size == 0
        ), (  # TODO: add a better check for word timestamps existence
            f"Unsupported language: {language}, use --batch_size to 0"
            " to generate word timestamps using whisper directly and fix this error."
        )
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append(
                    {"word": word[2], "start": word[0], "end": word[1]}
                )

    # Convert to mono for NeMo

    sound = AudioSegment.from_file(vocal_target).set_channels(1)
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

    # Diarize with NeMo MSSD

    msdd_model = NeuralDiarizer(
        cfg=create_config(temp_path, DOMAIN_TYPE="telephonic")
    ).to("cuda")
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Map Speakers to senteces with timestamps

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    # Realign speach using punctuation

    if language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Cleanup

    output_path = os.path.join("output", showname, episode)

    os.makedirs(output_path, exist_ok=True)

    os.remove("temp_outputs/mono_file.wav")

    os.rename("temp_outputs", output_path)

    path_textfile_with_speakers = os.path.join(
        output_path, f"{os.path.splitext(filename)[0]}.txt"
    )
    path_srtfile_with_speakers = os.path.join(
        output_path, f"{os.path.splitext(filename)[0]}.srt"
    )

    with open(path_textfile_with_speakers, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(path_srtfile_with_speakers, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    # Format results

    with open(path_textfile_with_speakers, "r") as f:
        lines = f.readlines()
        lines = [
            re.sub(" +", " ", line.strip("\ufeff").strip())
            for line in lines
            if line != "\n"
        ]

    with open(path_textfile_with_speakers, "w", encoding="utf-8-sig") as f:
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                f.write(f"{line}\n\n")
            else:
                f.write(f"{line}")


# Run Pipeline

results = getAllFiles()

for file in results:
    try:
        diarize(file)
        updateStatus(file, "done")
    except:
        updateStatus(file, "failed")
