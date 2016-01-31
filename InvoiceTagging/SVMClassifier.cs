using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace VoucherTagging
{
    public interface ISVMClassifier
    {
        void Load(Stream reader);
        void Save(Stream writer);
        void Save(string filePath);
        void Load(string filePath);
        void Initialize(Voucher[] vouchers);
        string Predict(List<string> ocrFeatures);
    }

    /// <summary>
    /// Support Vector Machine classification implementation
    /// </summary>
    public class SVMClassifier
    {
        FeatureManager _ftm = new FeatureManager { MaxWeightThreshold = 0.1 };
        Dictionary<string, SupportVectorMachine> _svm;

        #region Serialize

        void Read(BinaryReader binaryReader)
        {
            Func<BinaryReader, string> deserialize_string = (reader) =>
            {
                return reader.ReadString();
            };

            Func<BinaryReader, SupportVectorMachine> deserialize_svm = (reader) =>
            {
                SupportVectorMachine ret = SupportVectorMachine.Load(reader.BaseStream);
                return ret;
            };

            _svm = new Dictionary<string, SupportVectorMachine>();
            ReadFile<string, SupportVectorMachine>(binaryReader, _svm, deserialize_string, deserialize_svm);
        }

        static void ReadFile<TKey, TValue>(BinaryReader reader, Dictionary<TKey, TValue> dict, Func<BinaryReader, TKey> deserializeKey, Func<BinaryReader, TValue> deserializeValue)
        {
            int count = reader.ReadInt32();

            for (int i = 0; i < count; i++)
            {
                TKey key = deserializeKey(reader);
                TValue val = deserializeValue(reader);
                dict.Add(key, val);
            }
        }

        public void Write(BinaryWriter binaryWriter)
        {
            Action<string, BinaryWriter> serialize_string = (str, writer) =>
            {
                writer.Write(str);
            };

            Action<SupportVectorMachine, BinaryWriter> serialize_svm = (str, writer) =>
            {
                str.Save(writer.BaseStream);
            };

            WriteFile<string, SupportVectorMachine>(binaryWriter, _svm, serialize_string, serialize_svm);
        }

        static void WriteFile<TKey, TValue>(BinaryWriter writer, Dictionary<TKey, TValue> dict, Action<TKey, BinaryWriter> serializeKey, Action<TValue, BinaryWriter> serializeValue)
        {
            // Put count.
            writer.Write(dict.Count);
            // Write pairs.
            foreach (var pair in dict)
            {
                serializeKey(pair.Key, writer);
                serializeValue(pair.Value, writer);
            }
        }

        #endregion

        #region ISVMClassifier

        public void Save(string file)
        {
            using (FileStream fs = File.OpenWrite(file))
            using (BinaryWriter writer = new BinaryWriter(fs))
            {
                _ftm.Write(writer);
                Write(writer);
            }
        }

        public void Save(Stream stream)
        {
            using (var writer = new BinaryWriter(stream))
            {
                _ftm.Write(writer);
                Write(writer);
            }
        }

        public void Load(string file)
        {
            using (FileStream fs = File.OpenRead(file))
            using (BinaryReader reader = new BinaryReader(fs))
            {
                _ftm.Read(reader);
                Read(reader);
            }
        }

        public void Load(Stream stream)
        {
            using (BinaryReader reader = new BinaryReader(stream))
            {
                _ftm.Read(reader);
                Read(reader);
            }
        }

        public void Initialize(Voucher[] vouchers)
        {
            _ftm.Initialize(vouchers);
            TrainSVM(vouchers);
        }

        public string Predict(List<string> ocrFeatures)
        {
            int dim = _ftm.NumberOfValidFeatures + 1;
            double[] inputs2 = new double[dim];

            var features = _ftm.ReadWhenPredictingFeatures(ocrFeatures);
            features.ForEach(x =>
            {
                {
                    inputs2[x.Item1] = x.Item2;
                }
            });

            // clasify against each category
            var answers2 = _svm.Select(kvp => new Tuple<string, double>(kvp.Key, kvp.Value.Compute(inputs2))).OrderByDescending(x => x.Item2).ToArray();
            if (answers2.First().Item2 > -0.3)
            {
                return answers2.First().Item1;
            }
            else
            {
                return string.Empty;
            }
        }

        #endregion

        void TrainSVM(Voucher[] inputVouchers)
        {
            var vouchers = _ftm.ReadUniqueVouchers(inputVouchers);
            long iv = 0;
            int dim = _ftm.NumberOfValidFeatures + 1;
            double[][] inputs2 = new double[vouchers.Length][];
            string[] outputs2 = new string[vouchers.Length];

            Parallel.ForEach<Voucher>(vouchers,
                        (voucher) =>
                            {
                                long it = Interlocked.Increment(ref iv) - 1;
                                var features = _ftm.ReadWhenTrainingFeatures(voucher.OcrFeatures.ToList());
                                inputs2[it] = new double[dim];
                                features.ForEach(x =>
                                {
                                    inputs2[it][x.Item1] = x.Item2;
                                });
                                outputs2[it] = voucher.TagName;
                            });

            LibLinear(dim, inputs2, outputs2);
            return;
        }

        void LibLinear(int dim, double[][] inputs, string[] outputs)
        {
            _svm = outputs.Distinct().ToDictionary(x => x, x => new SupportVectorMachine(inputs: dim));

            Parallel.ForEach<KeyValuePair<string, SupportVectorMachine>>(_svm, // source collection
                                   (icat) => // method invoked by the loop on each iteration
                                   {
                                       var outputsDual = outputs.Select(x => x == icat.Key ? +1 : -1).ToArray();
                                       // Create a new linear-SVM for the problem dimensions
                                       var svm = icat.Value;
                                       // Create a learning algorithm for the problem's dimensions
                                       var teacher = new LinearDualCoordinateDescent(svm, inputs, outputsDual)
                                       {
                                           Loss = Loss.L2,
                                           // Complexity = 1000,
                                           //Tolerance = 1e-5
                                       };

                                       // Learn the classification
                                       double error = teacher.Run();
                                   });
        }
    }
}
