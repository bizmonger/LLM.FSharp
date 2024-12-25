namespace LLM.DataPreparation.EmbeddingLayer

open LLM.DataPreparation.Operations.Attention

module Compute =

    let score : Score.Compute =

        fun embedding1 embedding2 ->

            Array.empty

    let scores : Score.Compute =

        fun embedding1 embedding2 ->

            Array.empty