namespace LLM.DataPreparation

open System.Collections.Generic

module Language =

    // Tokens
    //---------------------------------------------
    type Text = string
    type BytePairEncoding = string
    type TokenedText = BytePairEncoding

    type Token  = int
    type Tokens = Token seq
    type Vocabulary = Dictionary<TokenedText,Token>

    // Vectors
    //---------------------------------------------
    type TokenVector   = int   array
    type ContextVector = float array
    type InputTokens   = TokenVector
    type ContentTokens = TokenVector
    type TargetTokens  = TokenVector
    type TokenVectors  = TokenVector array

    // Embeddings
    //---------------------------------------------
    type VectorEmbedding = float array
    type TokenEmbedding  = VectorEmbedding
    type TokenEmbeddings = VectorEmbedding array
    type EmbeddingsDictionary = Dictionary<Token, TokenEmbedding>

    // Input
    //---------------------------------------------
    type PositionalEmbedding  = VectorEmbedding
    type PositionalEmbeddings = VectorEmbedding array

    type InputEmbedding  = VectorEmbedding
    type InputEmbeddings = VectorEmbedding array

    type AttentionScore        = float array
    type AttentionWeight       = float array
    type AttentionWeights      = float array array
    type InputAttentionWeights = AttentionWeights

    // Training
    //---------------------------------------------
    type Weight  = float array
    type Weights = Weight array

    type QueryVector = float array
    type KeyVector   = float array
    type ValueVector = float array

    type QueryWeightParameters = Weights
    type KeyWeightParameters   = Weights
    type ValueWeightParameters = Weights

    // Input Target Pairs
    //---------------------------------------------
    type Prediction = TokenedText
    type Stride = int
    type ElementsPerRow = int
    type DimensionCount = int
    type MaxRowSize = int
    type BatchSize  = int