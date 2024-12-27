namespace LLM.DataPreparation.Attention

open LLM.DataPreparation.Operations.Attention

module Weights =

    let lookup : Lookup.TokenToWeights =

        fun token ->

            Array.empty