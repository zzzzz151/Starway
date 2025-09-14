# Starway

NN trainer for [Starlynn](https://github.com/zzzzz151/Starlynn) engine

# Usage (Linux only)

- Compile data converter and dataloader with `make converter` and `make dataloader`
    - Do not use compile flag `-DNDEBUG`, this repo is meant to be used with asserts enabled!

- Convert montyformat data to Starway format by running

    `./montyformat_to_starway <montyformat file> <output data file> <max RAM usage in MB> <batch offsets output file> <batch size> <batches to output>`

    and then

    `./interleave <input Starway data file> <output Starway data file> <converter's buffer capacity> <batch offsets input file> <batch offsets output file> <batch size>`

- Set training settings in `python/settings.py`

- Start training: run `python3 python/train.py`
    - Checkpoints are saved in `checkpoints` folder

- Optionally, quantize net: run `python3 python/quantize.py`

# Credits

[Monty](https://github.com/official-monty/Monty) project for the training data

[Montyformat docs](https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md)