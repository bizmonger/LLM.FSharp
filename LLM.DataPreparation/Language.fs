namespace LLM.DataPreparation

open System.Collections.Generic

module Language =

    type Text  = string
    type BytePairEncoding = string
    type TokenedText = BytePairEncoding

    type Token   = int
    type Tokens  = Token seq

    type Vocabulary    = Dictionary<TokenedText,Token>
    type TokenVector   = int array
    type InputTokens   = TokenVector
    type ContentTokens = TokenVector
    type TargetTokens  = TokenVector
    type TokenVectors  = TokenVector array

    type VectorEmbedding = float array
    type WeightMatrix    = VectorEmbedding array

    type TokenEmbedding  = VectorEmbedding
    type TokenEmbeddings = VectorEmbedding []

    type PositionalEmbedding  = VectorEmbedding
    type PositionalEmbeddings = VectorEmbedding []

    type InputEmbedding  = VectorEmbedding
    type InputEmbeddings = VectorEmbedding []

    type EmbeddingsDictionary = Dictionary<Token, TokenEmbedding>

    type Prediction = TokenedText
    type Stride = int
    type ElementsPerRow = int
    type DimensionCount = int
    type MaxRowSize = int
    type BatchSize  = int