module LLM.Transformer.Tests

open LLM.Transformer
open FsUnit

open NUnit.Framework

[<Test>]
let ``encode something`` () =
    
    Execute.encoder "some text goes here."
    |> should equal [||]

[<Test>]
let ``decode something`` () =

    Execute.decoder  "some text goes here." [||]
    |> should equal ""
