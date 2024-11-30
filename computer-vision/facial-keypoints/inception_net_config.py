inception_config1 = [
    # Inception Module 1 (Inception3a)
    {        
        'ch1x1': 64,
        'ch3x3_reduce': 96,
        'ch3x3': 128,
        'ch5x5_reduce': 16,
        'ch5x5': 32,
        'pool_proj': 32,
        'pool': False, # No pooling after this module
        'aux': False
        
    },
    # Inception Module 2 (Inception3b)
    {        
        'ch1x1': 128,
        'ch3x3_reduce': 128,
        'ch3x3': 192,
        'ch5x5_reduce': 32,
        'ch5x5': 96,
        'pool_proj': 64,
        'pool': True,  # Pooling after this module (pool3)
        'aux': False
    },
    # Inception Module 3 (Inception4a)
    {        
        'ch1x1': 192,
        'ch3x3_reduce': 96,
        'ch3x3': 208,
        'ch5x5_reduce': 16,
        'ch5x5': 48,
        'pool_proj': 64,
        'pool': False,
        'aux': True,
        'aux_in_channels': 512
    },
    # Inception Module 4 (Inception4b)
    {
        'ch1x1': 160,
        'ch3x3_reduce': 112,
        'ch3x3': 224,
        'ch5x5_reduce': 24,
        'ch5x5': 64,
        'pool_proj': 64,
        'pool': False,
        'aux': False
    },
    # Inception Module 5 (Inception4c)
    {
        'ch1x1': 128,
        'ch3x3_reduce': 128,
        'ch3x3': 256,
        'ch5x5_reduce': 24,
        'ch5x5': 64,
        'pool_proj': 64,
        'pool': False,
        'aux': False
    },
    # Inception Module 6 (Inception4d)
    {
        'ch1x1': 112,
        'ch3x3_reduce': 144,
        'ch3x3': 288,
        'ch5x5_reduce': 32,
        'ch5x5': 64,
        'pool_proj': 64,
        'pool': False,
        'aux': True,
        'aux_in_channels': 528
    },
    # Inception Module 7 (Inception4e)
    {
        'ch1x1': 256,
        'ch3x3_reduce': 160,
        'ch3x3': 320,
        'ch5x5_reduce': 32,
        'ch5x5': 128,
        'pool_proj': 128,
        'pool': True,  # Pooling after this module (pool4),
        'aux': False
    },
    # Inception Module 8 (Inception5a)
    {
        'ch1x1': 256,
        'ch3x3_reduce': 160,
        'ch3x3': 320,
        'ch5x5_reduce': 32,
        'ch5x5': 128,
        'pool_proj': 128,
        'pool': False,
        'aux': False
    },
    # Inception Module 9 (Inception5b)
    {
        'ch1x1': 384,
        'ch3x3_reduce': 192,
        'ch3x3': 384,
        'ch5x5_reduce': 48,
        'ch5x5': 128,
        'pool_proj': 128,
        'pool': False,
        'aux': False
    },
]

inception_config2 = [
    # Inception Module 1
    {
        'ch1x1': 96,
        'ch3x3_reduce': 144,
        'ch3x3': 192,
        'ch5x5_reduce': 24,
        'ch5x5': 48,
        'pool_proj': 48,
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 2
    {
        'ch1x1': 192,
        'ch3x3_reduce': 192,
        'ch3x3': 288,
        'ch5x5_reduce': 48,
        'ch5x5': 144,
        'pool_proj': 96,
        'pool': True,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 3
    {
        'ch1x1': 288,
        'ch3x3_reduce': 144,
        'ch3x3': 312,
        'ch5x5_reduce': 24,
        'ch5x5': 72,
        'pool_proj': 96,
        'pool': False,
        'aux': True,
        'aux_in_channels': 768
    },
    # Inception Module 4
    {
        'ch1x1': 240,
        'ch3x3_reduce': 168,
        'ch3x3': 336,
        'ch5x5_reduce': 36,
        'ch5x5': 96,
        'pool_proj': 96,
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 5
    {
        'ch1x1': 192,
        'ch3x3_reduce': 192,
        'ch3x3': 384,
        'ch5x5_reduce': 36,
        'ch5x5': 96,
        'pool_proj': 96,
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 6
    {
        'ch1x1': 168,
        'ch3x3_reduce': 216,
        'ch3x3': 432,
        'ch5x5_reduce': 48,
        'ch5x5': 96,
        'pool_proj': 96,
        'pool': False,
        'aux': True,
        'aux_in_channels': 792
    },
    # Inception Module 7
    {
        'ch1x1': 384,
        'ch3x3_reduce': 240,
        'ch3x3': 480,
        'ch5x5_reduce': 48,
        'ch5x5': 192,
        'pool_proj': 192,
        'pool': True,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 8
    {
        'ch1x1': 384,
        'ch3x3_reduce': 240,
        'ch3x3': 480,
        'ch5x5_reduce': 48,
        'ch5x5': 192,
        'pool_proj': 192,
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 9
    {
        'ch1x1': 576,
        'ch3x3_reduce': 288,
        'ch3x3': 576,
        'ch5x5_reduce': 72,
        'ch5x5': 192,
        'pool_proj': 192,
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
]

inception_config3_complex = [
    # Inception Module 1
    {
        'ch1x1': 135,            # Increased from 96
        'ch3x3_reduce': 201,     # Increased from 144
        'ch3x3': 269,            # Increased from 192
        'ch5x5_reduce': 34,      # Increased from 24
        'ch5x5': 67,             # Increased from 48
        'pool_proj': 67,         # Increased from 48
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 2
    {
        'ch1x1': 269,            # Increased from 192
        'ch3x3_reduce': 269,     # Increased from 192
        'ch3x3': 403,            # Increased from 288
        'ch5x5_reduce': 67,      # Increased from 48
        'ch5x5': 202,            # Increased from 144
        'pool_proj': 135,        # Increased from 96
        'pool': True,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 3
    {
        'ch1x1': 403,            # Increased from 288
        'ch3x3_reduce': 201,     # Increased from 144
        'ch3x3': 437,            # Increased from 312
        'ch5x5_reduce': 34,      # Increased from 24
        'ch5x5': 101,            # Increased from 72
        'pool_proj': 135,        # Increased from 96
        'pool': False,
        'aux': True,
        'aux_in_channels': 1076  # Increased from 768
    },
    # Inception Module 4
    {
        'ch1x1': 336,            # Increased from 240
        'ch3x3_reduce': 235,     # Increased from 168
        'ch3x3': 470,            # Increased from 336
        'ch5x5_reduce': 50,      # Increased from 36
        'ch5x5': 135,            # Increased from 96
        'pool_proj': 135,        # Increased from 96
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 5
    {
        'ch1x1': 269,            # Increased from 192
        'ch3x3_reduce': 269,     # Increased from 192
        'ch3x3': 538,            # Increased from 384
        'ch5x5_reduce': 50,      # Increased from 36
        'ch5x5': 135,            # Increased from 96
        'pool_proj': 135,        # Increased from 96
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 6
    {
        'ch1x1': 235,            # Increased from 168
        'ch3x3_reduce': 302,     # Increased from 216
        'ch3x3': 605,            # Increased from 432
        'ch5x5_reduce': 67,      # Increased from 48
        'ch5x5': 135,            # Increased from 96
        'pool_proj': 135,        # Increased from 96
        'pool': False,
        'aux': True,
        'aux_in_channels': 1110  # Increased from 792
    },
    # Inception Module 7
    {
        'ch1x1': 538,            # Increased from 384
        'ch3x3_reduce': 336,     # Increased from 240
        'ch3x3': 672,            # Increased from 480
        'ch5x5_reduce': 67,      # Increased from 48
        'ch5x5': 269,            # Increased from 192
        'pool_proj': 269,        # Increased from 192
        'pool': True,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 8
    {
        'ch1x1': 538,            # Increased from 384
        'ch3x3_reduce': 336,     # Increased from 240
        'ch3x3': 672,            # Increased from 480
        'ch5x5_reduce': 67,      # Increased from 48
        'ch5x5': 269,            # Increased from 192
        'pool_proj': 269,        # Increased from 192
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
    # Inception Module 9
    {
        'ch1x1': 807,            # Increased from 576
        'ch3x3_reduce': 403,     # Increased from 288
        'ch3x3': 807,            # Increased from 576
        'ch5x5_reduce': 101,     # Increased from 72
        'ch5x5': 269,            # Increased from 192
        'pool_proj': 269,        # Increased from 192
        'pool': False,
        'aux': False,
        'aux_in_channels': 0
    },
]

