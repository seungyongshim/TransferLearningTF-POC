using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TransferLearningTF_POC
{
    internal class Program
    {
        private static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        private static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        private static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
        private static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
        private static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
        private static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");

        public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };

            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine("=============== Making single image classification ===============");
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
        }

        // Build and train model
        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                            .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                                ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                            .AppendCacheCheckpoint(mlContext);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            Console.WriteLine("=============== Training classification model ===============");
            ITransformer model = pipeline.Fit(trainingData);

            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            DisplayResults(imagePredictionData);

            Console.WriteLine("=============== Classification metrics ===============");

            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(predictions,
                  labelColumnName: "LabelKey",
                  predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            //Need to parse through the tags.tsv file to combine the file path to the
            // image name for the ImagePath property so that the image file can be found.

            return File.ReadAllLines(file)
             .Select(line => line.Split('\t'))
             .Select(line => new ImageData()
             {
                 ImagePath = Path.Combine(folder, line[0])
             });
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }

        private static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            ITransformer model = GenerateModel(mlContext);

            ClassifySingleImage(mlContext, model);

            Console.ReadKey();
        }

        private struct InceptionSettings
        {
            public const bool ChannelsLast = true;
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
        }

        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;
        }

        public class ImagePrediction : ImageData
        {
            public string PredictedLabelValue;
            public float[] Score;
        }
    }
}