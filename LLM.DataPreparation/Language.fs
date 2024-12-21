namespace LLM.DataPreparation

module Language =

    type Text  = string
    type BytePairEncoding = string
    type TokenedText = BytePairEncoding

    type Token   = int
    type Tokens  = Token seq
    type Vocabulary = Map<Token,TokenedText>
    type Vector  = (int array)
    type Vectors = Vector seq
    type Embedding  = Vectors

    type Prediction = TokenedText