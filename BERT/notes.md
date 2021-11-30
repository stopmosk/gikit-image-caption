Normal text, **bold text**, *semibold text*, __bold text 1__, _semibold text1_

# max seq lengh  = 70
The maximum total input sequence length after tokenization. 
Sequences longer than this will be truncated, sequences shorter will be padded.

# max seq a length  = 40
The maximum sequence length for caption.

# max img seq length  = 50
The maximum total input image sequence length.

# max gen length  = 20
Max length of generated sentences.


# SCST train
    max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length

# Eval
    max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length

# SCST or Eval
    max_seq_length = args.max_gen_length + max_od_labels_len

       [[ 101, 2116, 2367, 4127, 1997, 4683,  103, 1037, 2395, 1012,  103,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0, 4316, 2455, 4316, 3392, 4316, 2455, 4316, 4316,
         2482, 4316, 2455, 4316, 4316, 2482, 2482, 2455, 4316, 2455, 4316, 4316,
         4316, 5217, 2455, 4316, 2482, 2455, 4316, 2482, 5217,  102]], - input_ids (1, 70)

      ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], - token_type_values (1, 70)

      ([[[1, 0, 0,  ..., 1, 1, 1],
         [1, 1, 0,  ..., 1, 1, 1],
         [1, 1, 1,  ..., 1, 1, 1],
         ...,
         [0, 0, 0,  ..., 1, 1, 1],
         [0, 0, 0,  ..., 1, 1, 1],
         [0, 0, 0,  ..., 1, 1, 1]]], attention_mask (1, 120, 120)

      ([[[0.0000, 0.0489, 0.0000,  ..., 0.7337, 0.2361, 0.5652],
         [0.3639, 6.7504, 0.0000,  ..., 0.9944, 0.2076, 0.1644],
         [0.0860, 0.0176, 0.0000,  ..., 0.9327, 0.4879, 0.7451],
         ...,
         [0.3376, 0.6810, 0.0000,  ..., 0.9983, 0.9248, 0.5228],
         [0.4059, 0.0000, 0.0000,  ..., 0.8050, 0.0433, 0.0665],
         [0.0278, 0.0000, 0.0000,  ..., 0.5275, 0.0186, 0.0676]]], - img_feats (1, 50, 2054)

      ([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], - masked_pos (1, 70)

      ([[2006,  102,    0]], device='cuda:0') - masked_ids


