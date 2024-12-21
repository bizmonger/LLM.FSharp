namespace LLM.DataPreparation

open System.Collections.Generic

module Language =

    type Text  = string
    type BytePairEncoding = string
    type TokenedText = BytePairEncoding

    type Token   = int
    type Tokens  = Token seq
    type Vocabulary   = Dictionary<Token,TokenedText>
    type Vector       = (int array)
    type InputTokens  = Vector
    type TargetTokens = Vector
    type Vectors      = Vector seq
    type Embedding    = Vectors

    type Prediction = TokenedText
    type Stride = int