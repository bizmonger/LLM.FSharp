module LLM.Transformer.Tests

open LLM.DataPreparation
open FsUnit

open NUnit.Framework

[<Test>]
let ``encode something`` () =
    
    Tokenizer.encode "some text goes here."
    |> should equal [||]

[<Test>]
let ``decode something`` () =

    Tokenizer.decode [||]
    |> should equal ""
