namespace LLM.DataPreparation

open System.Collections.Generic

module Language =

    type Text  = string
    type BytePairEncoding = string
    type TokenedText = BytePairEncoding

    type Token   = int
    type Tokens  = Token seq
    type Vocabulary   = Dictionary<Token,TokenedText>
    type TokenVector  = int array
    type InputTokens  = TokenVector
    type TargetTokens = TokenVector
    type TokenVectors = TokenVector array

    type VectorEmbedding = float array
    type WeightMatrix    = VectorEmbedding array

    type TokenEmbedding       = VectorEmbedding
    type TokenEmbeddings      = VectorEmbedding []

    type PositionalEmbedding  = VectorEmbedding
    type PositionalEmbeddings = VectorEmbedding []

    type InputEmbedding       = VectorEmbedding
    type InputEmbeddings      = VectorEmbedding []

    type Prediction = TokenedText
    type Stride = int
    type DimensionCount = int
    type MaxRowSize = int
    type BatchSize  = int