﻿module LLM.DataPreparation.Tests

open System
open FsUnit
open NUnit.Framework
open LLM.DataPreparation.Language
open LLM.DataPreparation.EmbeddingLayer
open LLM.DataPreparation.EmbeddingLayer.WeightMatrix
open LLM.DataPreparation.EmbeddingLayer.Compute

[<Test>]
let ``build vocabulary`` () =

    // Test
    let vocabulary = DataSource.createVocabulary "Here's some content."

    // Verify
    vocabulary.Count |> should be (greaterThanOrEqualTo 4)

//[<Test>]
//let ``encode something`` () =

//    // Setup
//    let content    = "First of all, some text goes here."
//    let vocabulary = content |> DataSource.createVocabulary
//    let textInput  = "First of all"
    
//    vocabulary 
//    |> Tokenizer.encode textInput
//    |> Array.length
//    |> should be (greaterThan 3)

[<Test>]
let ``get input/target pairs`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let vectorSize = 4
    let stride = 1
    let contentTokens = content |> Tokenizer.extractTokens vocabulary

    // Test
    let input, target  = Tokenizer.inputTargetPairs contentTokens vectorSize stride
    let textDictionary = vocabulary |> Tokenizer.toTextDictionary

    let inputText  = input .[0] |> Array.map(fun token -> textDictionary.[token])
    let targetText = target.[0] |> Array.map(fun token -> textDictionary.[token])

    inputText  |> should equal [|"First"; "of"; "all"; ","|]
    targetText |> should equal [| "of"; "all"; ","; "some";|]

[<Test>]
let ``initialize token embedding`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4

    // Test
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions 

    // Verify
    embeddingsDictionary.Count |> should be (greaterThanOrEqualTo 9)


[<Test>]
let ``Retrieve a token embedding`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions 
    let token = 0

    // Test
    let embedding = embeddingsDictionary.[token]

    // Verify
    embedding |> Array.isEmpty |> should equal false

[<Test>]
[<Ignore("Ignore a test")>]
let ``Add positional encoding to token`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions 
    let token = 3

    // Test
    ()

    // Verify
    Assert.Fail()

[<Test>]
[<Ignore("Ignore a test")>]
let ``Calculate attention scores`` () =

    // Setup
    let content    = "Your journey starts with one step. Each additonal step is one step closer to your destination."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions

    let textInput   = "Your journey starts with one step"
    let inputTokens = textInput |> Tokenizer.extractTokens vocabulary

    let tokenEmbeddings      = inputTokens     |> Tokens.toTokenEmbeddings embeddingsDictionary
    let positionalEmbeddings = inputTokens     |> Tokens.toPositionalEmbeddings dimensions
    let inputEmbeddings      = tokenEmbeddings |> Tokens.toInputEmbeddings positionalEmbeddings

    let queryItem = inputEmbeddings.[1]
    
    // Test
    let scores = queryItem |> Compute.attentionScores inputEmbeddings

    // Verify
    Assert.Fail()
    
[<Test>]
let ``Calculate vector product with equal size`` () =

    // Setup
    let vectorA = [|1.0;2.0|]
    let vectorB = [|2.0;3.0|]

    // Test
    let vectorProduct = Compute.vectorProduct vectorA vectorB

    // Verify
    vectorProduct.[0] |> should equal 2.0
    vectorProduct.[1] |> should equal 6.0

[<Test>]
let ``Calculate vector product with non equal size`` () =

    // Setup
    let vectorA = [|0.3155|]
    let vectorB = [|0.2;0.5|]

    // Test
    let vectorProduct = Compute.vectorProduct vectorA vectorB

    // Verify
    vectorProduct |> should equal [|0.0631; 0.15775|]

[<Test>]
let ``Calculate dot product`` () =

    // Setup
    let vectorA = [|1.0;2.0|]
    let vectorB = [|2.0;3.0|]

    // Test
    let dotProduct = Compute.vectorMultiplicationAndSum vectorA vectorB

    // Verify
    dotProduct |> should equal 8

[<Test>]
let ``Calculate attention score`` () =

    // Setup
    let vectorInput1 = [|0.43;0.15;0.89|]
    let vectorQuery  = [|0.55;0.87;0.66|]

    // Test
    let score = Compute.vectorMultiplicationAndSum vectorInput1 vectorQuery

    // Verify
    score |> should equal 0.9544

[<Test>]
let ``Calculate attention weight`` () =

    // Setup
    let inputEmbeddings = [|
                            [|0.43;0.15;0.89|]
                            [|0.55;0.87;0.66|]
                            [|0.57;0.85;0.64|]
                            [|0.22;0.58;0.33|]
                            [|0.77;0.25;0.10|]
                            [|0.05;0.80;0.55|]
                          |]

    let vectorQuery = inputEmbeddings.[1]

    let scores = Compute.attentionScores inputEmbeddings vectorQuery

    // Test
    let weights = Compute.attentionWeights scores
    let sum = weights |> Array.sum

    // Verify
    let rounded = weights |> Array.map(fun w -> Math.Round(w,4))

    rounded |> should equal [|Math.Round(0.1385,4);
                              Math.Round(0.2379,4);
                              Math.Round(0.2333,4);
                              Math.Round(0.1240,4);
                              Math.Round(0.1082,4);
                              Math.Round(0.1581,4);
                            |]

[<Test>]
let ``Normalize attention weights`` () =

    // Setup
    let inputEmbeddings = [|
                            [|0.43;0.15;0.89|]
                            [|0.55;0.87;0.66|]
                            [|0.57;0.85;0.64|]
                            [|0.22;0.58;0.33|]
                            [|0.77;0.25;0.10|]
                            [|0.05;0.80;0.55|]
                          |]

    let vectorQuery = [|0.55;0.87;0.66|]

    let scores  = Compute.attentionScores inputEmbeddings vectorQuery
    
    // Test
    let weights = Compute.attentionWeights (scores)
    let sum = weights |> Array.sum

    // Verify
    weights |> Array.length |> should equal 6
    sum |> should equal 1

[<Test>]
let ``Calculate context vector - 2`` () =

    // Setup
    let inputEmbeddings = [|
                            [|0.2;0.5|]
                            [|0.8;0.1|]
                            [|0.3;0.9|]
                          |]

    let x1_query   = inputEmbeddings.[0]
    let x1_scores  = Compute.attentionScores inputEmbeddings x1_query
    let x1_weights = Compute.attentionWeights x1_scores

    // Test
    let x1_embeddingWeight = ProductOf.numberAndVector x1_weights.[0] inputEmbeddings.[0]

    // Verify
    Math.Round(x1_embeddingWeight.[0], 4) |> should equal 0.0631
    Math.Round(x1_embeddingWeight.[1], 4) |> should equal 0.1578

[<Test>]
let ``Calculate context vector`` () =

    // Setup
    let content = "The cat sat on the hat"
    let vocabulary  = content    |> DataSource.createVocabulary
    let tokenToText = vocabulary |> DataSource.toTokenLookup

    let embeddingsDictionary = EmbeddingsDictionary()
    embeddingsDictionary.Add(vocabulary.["The"],[|0.2;0.5|])
    embeddingsDictionary.Add(vocabulary.["cat"],[|0.8;0.1|])
    embeddingsDictionary.Add(vocabulary.["sat"],[|0.3;0.9|])

    let inputText = "The cat sat"

    // Test
    let contextVectorDetails = inputText |> Tokenizer.extractTokens vocabulary 
                                         |> Array.map(fun t -> embeddingsDictionary.[t])
                                         |> Compute.contextVectorDetails tokenToText

    // Verify
    (contextVectorDetails.Head).ContextVector.[0] |> should equal 0.41408524643709554
    (contextVectorDetails.Head).ContextVector.[1] |> should equal 0.54076247503470021

[<Test>]
let ``Compute vector sum`` () =

    // Setup
    let tokenEmbedding      = [|1.0;1.0;1.0|]
    let positionalEmbedding = [|1.1;1.2;1.3|]

    // Test
    let inputEmbeddings = Compute.vectorSum tokenEmbedding positionalEmbedding

    // Verify
    inputEmbeddings |> should equal [|2.1;2.2;2.3|]

[<Test>]
let ``Computing matrix multiplication resilts in correct row count and column count of product matrix``() =

    // Setup
    let matrixA = Array2D.init 2 3 (fun _ _ -> 0.0)
    let matrixB = Array2D.init 3 2 (fun _ _ -> 0.0)

    // Test
    let matrixC = Compute.matrixMultiplication matrixA matrixB

    // Verify
    matrixC |> Array2D.length1 |> should equal 2
    matrixC |> Array2D.length2 |> should equal 2

[<Test>]
let ``Compute matrix multiplication``() =

    // Setup
    let matrixA = Array2D.init 2 3 (fun _ _ -> 0.0)
    let matrixB = Array2D.init 3 2 (fun _ _ -> 0.0)

    // Test
    let matrixC = Compute.matrixMultiplication matrixA matrixB

    // Verify
    failwith "TODO"

[<Test>]
let ``Calculate input embedding`` () =

    // Setup
    let tokenEmbeddings = [|
                           [|1.0;1.0;1.0|]
                           [|1.0;1.0;1.0|]
                         |]

    let positionalEmbeddings = [|
                                [|1.1;1.2;1.3|]
                                [|2.1;2.2;2.3|]
                               |]

    // Test
    let inputEmbeddings = Tokens.toInputEmbeddings tokenEmbeddings positionalEmbeddings

    // Verify
    inputEmbeddings.[0] |> should equal [|2.1;2.2;2.3|]
    inputEmbeddings.[1] |> should equal [|3.1;3.2;3.3|]

[<Test>]
let ``Calculate context vector - 3`` () =

    // Setup
    let content = "Your journey starts with one step. This step can be steep. But it can end with success if commited from start to finish."
    let vocabulary  = content    |> DataSource.createVocabulary
    let tokenToText = vocabulary |> DataSource.toTokenLookup

    let embeddingsDictionary = EmbeddingsDictionary()
    embeddingsDictionary.Add(vocabulary.["Your"]   ,[|0.43;0.15;0.89|])
    embeddingsDictionary.Add(vocabulary.["journey"],[|0.55;0.87;0.66|])
    embeddingsDictionary.Add(vocabulary.["starts"] ,[|0.57;0.85;0.64|])
    embeddingsDictionary.Add(vocabulary.["with"]   ,[|0.22;0.58;0.33|])
    embeddingsDictionary.Add(vocabulary.["one"]    ,[|0.77;0.25;0.10|])
    embeddingsDictionary.Add(vocabulary.["step"]   ,[|0.05;0.80;0.55|])

    let inputText = "Your journey starts with one step"

    // Test
    let contextVectorDetails = inputText |> Tokenizer.extractTokens vocabulary 
                                         |> Array.map(fun t -> embeddingsDictionary.[t])
                                         |> Compute.contextVectorDetails tokenToText
    
    // Verify
    let contextVectors = contextVectorDetails |> Array.ofList |> Array.map(fun v -> v.ContextVector)
    (contextVectors.[1]) |> should equal [|0.44186574785129207; 0.6514819780302219; 0.56830888772572918|]

[<Test>]
let ``Compute weight key vector`` () =

    // Setup
    let content = "Your journey starts with one step. This step can be steep. But it can end with success if commited from start to finish."
    let vocabulary  = content    |> DataSource.createVocabulary
    let tokenToText = vocabulary |> DataSource.toTokenLookup

    let embeddingsDictionary = EmbeddingsDictionary()
    embeddingsDictionary.Add(vocabulary.["Your"]   ,[|0.43;0.15;0.89|])
    embeddingsDictionary.Add(vocabulary.["journey"],[|0.55;0.87;0.66|])
    embeddingsDictionary.Add(vocabulary.["starts"] ,[|0.57;0.85;0.64|])
    embeddingsDictionary.Add(vocabulary.["with"]   ,[|0.22;0.58;0.33|])
    embeddingsDictionary.Add(vocabulary.["one"]    ,[|0.77;0.25;0.10|])
    embeddingsDictionary.Add(vocabulary.["step"]   ,[|0.05;0.80;0.55|])

    let textInput  = "Your journey starts with one step"
    let tokenInput = textInput |> Tokenizer.extractTokens vocabulary

    let contextVectorDetails = textInput |> Tokenizer.extractTokens vocabulary 
                                         |> Array.map(fun t -> embeddingsDictionary.[t])
                                         |> Compute.contextVectorDetails tokenToText
    
    let weightQueryMatrix = Initialize.matrix 3 3
    let weightKeyMatrix   = Initialize.matrix 3 3
    let weightValueMatrix = Initialize.matrix 3 3

    let x_2 = embeddingsDictionary.[1]

    // Test
    let values  = weightQueryMatrix.Values |> Array.ofSeq
    //let query_2 = x_2 |> Compute.vectorProduct weightQueryMatrix.Values

    // Verify
    weightQueryMatrix |> Seq.length |> should equal 3