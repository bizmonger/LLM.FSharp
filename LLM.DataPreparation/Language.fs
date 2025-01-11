namespace LLM.DataPreparation

open System.Collections.Generic

module Language =

    // Tokens
    //---------------------------------------------
    type Text = string
    type BytePairEncoding = string
    type TokenedText = BytePairEncoding
    type DotProduct  = float
    type Matrix      = float array array

    type Token = int
    type RelativeToken = int
    type EmbeddingIndex = int
    type Tokens = Token array
    type Vocabulary  = Dictionary<TokenedText,Token>
    type TokenToText = Dictionary<Token,TokenedText>

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

    type AttentionScore        = float
    type AttentionScores       = float array
    type AttentionWeight       = float
    type AttentionWeights      = float array
    type InputAttentionWeights = AttentionWeights

    type QueryProduct = {
        Text           : string
        Token          : int
        InputEmbedding : float array
        Scores         : float array
        Weights        : float array
        ContextVector  : float array
    }

    type WeightAndInputEmbedding = {
        Weight : float
        InputEmbedding : float array
    }

    type WeightAndInputEmbeddingProduct = {
        WeightAndInputEmbedding: WeightAndInputEmbedding
        Product : float array
    }

    // Training
    //---------------------------------------------
    type WeightParameters = float array
    type WeightParametersMatrix = WeightParameters array

    type QueryVector = float array
    type KeyVector   = float array
    type ValueVector = float array

    type QueryWeightParameters = WeightParametersMatrix
    type KeyWeightParameters   = WeightParametersMatrix
    type ValueWeightParameters = WeightParametersMatrix

    // Input Target Pairs
    //---------------------------------------------
    type Prediction = TokenedText
    type Stride = int
    type ElementsPerRow = int
    type DimensionCount = int
    type MaxRowSize = int
    type BatchSize  = int