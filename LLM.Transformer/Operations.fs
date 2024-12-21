namespace LLM.Transformer

open Language

module Operations =

    type Encode = Text -> Vectors

    type Decode = Text -> Vectors -> Text