namespace LLM.DataPreparation

open LLM.DataPreparation.Operations

module Tokenizer =

    let encode : Text.ToEmbedding =

        fun textInput vocabulary ->

            textInput
        
            [||]

    let decode : Vectors.PredictToken =

        fun _ -> "TODO"