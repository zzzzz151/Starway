# Starway

NN trainer for [Starlynn](https://github.com/zzzzz151/Starlynn) engine

# Usage

- Compile data converter and dataloader with `make converter` and `make dataloader`
    - Do not use compile flag `-DNDEBUG`, this repo is meant to be used with asserts enabled!

- Convert montyformat data to Starway format by running

    ```
    ./montyformat_to_starway[.exe]
        <montyformat file>
        <output data file>
        <batch size>
        <batches to output>
    ```

- Set training settings in `python/settings.py`

- Start training: run `python3 python/train.py`
    - Checkpoints are saved in `checkpoints` folder

- Export a net checkpoint to binary file: run `python3 python/net_to_bin.py` (optionally quantizes)

# Credits

[Monty](https://github.com/official-monty/Monty) project for the training data

[Montyformat docs](https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md)