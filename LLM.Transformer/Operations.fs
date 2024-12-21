namespace LLM.Transformer

open LLM.DataPreparation.Language

module Operations =

    type Encode = Text -> Vectors

    type Decode = Text -> Vectors -> Prediction

    type toText = Token -> Text