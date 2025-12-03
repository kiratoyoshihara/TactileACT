class Config:
    # Data dimensions
    # KUKA iiwa (7 joints) + Gripper (1 joint) = 8
    STATE_DIM = 8
    ACTION_DIM = 8
    
    # Camera resolution (Resized for ResNet)
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    
    # Model Hyperparameters
    HIDDEN_DIM = 512
    ENC_LAYERS = 4
    DEC_LAYERS = 7
    DIM_FEEDFORWARD = 3200
    NHEADS = 8
    DROPOUT = 0.1
    
    # CVAE Hyperparameters
    LATENT_DIM = 32
    
    # Action Chunking
    CHUNK_SIZE = 100  # Number of future actions to predict
    
    # Training Hyperparameters
    BATCH_SIZE = 8 
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    KL_WEIGHT = 10.0  # Beta for CVAE KL Loss
    
    # trick: Modality Dropout
    # Probability of dropping vision to force tactile usage
    MODALITY_DROPOUT_RATE = 0.3