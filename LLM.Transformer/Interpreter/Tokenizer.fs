namespace LLM.Transformer

open LLM.DataPreparation.Operations

module Tokenizer =

    let encode : Text.ToEmbedding =

        fun _ -> [||]

    let decode : Vectors.NextTokenPrediction =

        fun _ -> "TODO"