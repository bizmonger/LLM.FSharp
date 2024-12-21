namespace LLM.DataPreparation

module Language =

    type Text  = string
    type Word  = string
    type TokenedText = string

    type Token   = int
    type Tokens  = Token seq
    type Vocabulary = Map<Token,TokenedText>
    type Vector  = (int array)
    type Vectors = Vector seq
    type Embedding  = Vectors

    type Prediction = Word
    