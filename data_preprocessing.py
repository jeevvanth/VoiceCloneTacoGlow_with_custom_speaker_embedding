import os
import glob
import soundfile as sf

def prepare_librispeech_dev_clean(src_root="LibriSpeech/dev-clean", dst_root="data"):
    """
    Converts LibriSpeech/dev-clean into speakerX_Y.wav format for GE2E loss training.
    """
    os.makedirs(dst_root, exist_ok=True)

    speaker_dirs = sorted(glob.glob(os.path.join(src_root, "*")))
    speaker_index = 1

    for spk_dir in speaker_dirs:
        chapter_dirs = sorted(glob.glob(os.path.join(spk_dir, "*")))
        file_index = 1

        for chapter_dir in chapter_dirs:
            flac_files = sorted(glob.glob(os.path.join(chapter_dir, "*.flac")))

            for flac_file in flac_files:
                
                audio, sr = sf.read(flac_file)

                
                new_filename = f"speaker{speaker_index}_{file_index}.wav"
                dst_path = os.path.join(dst_root, new_filename)

                # Save as .wav
                sf.write(dst_path, audio, sr)
                file_index += 1

        speaker_index += 1

    print(f" Finished processing. {speaker_index - 1} speakers converted.")


if __name__ == "__main__":
    prepare_librispeech_dev_clean("LibriSpeech/dev-clean", "data")
