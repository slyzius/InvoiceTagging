using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VoucherTagging
{
    class VoucherFeatures : IEquatable<VoucherFeatures>
    {
        public List<Tuple<int, double>> Features { get; set; }

        #region IEquatable

        public bool Equals(VoucherFeatures other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;

            if (other.Features.Count == Features.Count)
            {
                for (int i = 0; i < Features.Count && i < other.Features.Count; i++)
                {
                    if (other.Features[i].Item1 != Features[i].Item1)
                    {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        #endregion

        #region object

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            return obj is VoucherFeatures && Equals((VoucherFeatures)obj);
        }

        public override int GetHashCode()
        {
            int hash = Features.Count;
            for (int i = 0; i < Features.Count; ++i)
            {
                hash = unchecked(hash * 314159 + Features[i].Item1);
            }
            return hash;
        }


        #endregion

    }

    public interface IFeatureManager
    {
        void Read(BinaryReader reader);
        void Write(BinaryWriter writer);
        void Initialize(IEnumerable<Voucher> vouchers);
        Voucher[] ReadUniqueVouchers(Voucher[] vouchers);
        List<Tuple<int, double>> ReadWhenTrainingFeatures(List<string> ocrFeatures);
        List<Tuple<int, double>> ReadWhenPredictingFeatures(List<string> ocrFeatures);
    }

    public class FeatureManager : IFeatureManager
    {
        Dictionary<int, int> _filteredFeatures = new Dictionary<int, int>();
        Dictionary<int, double> _featureWeight = new Dictionary<int, double>();
        Dictionary<int, string> _featureById = new Dictionary<int, string>();
        Dictionary<string, Dictionary<int, int>> _bgrams = new Dictionary<string, Dictionary<int, int>>();
        int _ngram = 3;
        double _similiarityThreshold = 0.5;
        double _maxWeightThreshold;
        double _weigthExpValue;
        public double MaxWeightThreshold { get { return _maxWeightThreshold; } set { _maxWeightThreshold = value; _weigthExpValue = Math.Log(0.01) / value; } }
        int _numberOfValidFeatures = 0;

        /// <summary>
        /// Return number of relevant features. We ommit some features if their importance is low, e.g. the feature cannot
        /// uniqually represent the vouchers category: moms, faktura,...
        /// </summary>
        public int NumberOfValidFeatures
        {
            get
            {
                if (_numberOfValidFeatures != 0)
                {
                    return _numberOfValidFeatures;
                }
                else
                {
                    _numberOfValidFeatures = _featureWeight.Where(x => x.Value >= 0.01).Count();
                    return _numberOfValidFeatures;
                }
            }
        }

        #region serialization

        public void Read(BinaryReader binaryReader)
        {
            Func<BinaryReader, int> deserialize_int = (reader) =>
            {
                return reader.ReadInt32();
            };

            Func<BinaryReader, double> deserialize_double = (reader) =>
            {
                return reader.ReadDouble();
            };

            Func<BinaryReader, string> deserialize_string = (reader) =>
            {
                return reader.ReadString();
            };

            Func<BinaryReader, Dictionary<int, int>> deserialize_dictionary_int_int = (reader) =>
            {
                Dictionary<int, int> ret = new Dictionary<int, int>();
                ReadFile<int, int>(reader, ret, deserialize_int, deserialize_int);
                return ret;
            };

            ReadFile<int, int>(binaryReader, _filteredFeatures, deserialize_int, deserialize_int);
            ReadFile<int, double>(binaryReader, _featureWeight, deserialize_int, deserialize_double);
            ReadFile<int, string>(binaryReader, _featureById, deserialize_int, deserialize_string);
            ReadFile<string, Dictionary<int, int>>(binaryReader, _bgrams, deserialize_string, deserialize_dictionary_int_int);
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
            Action<int, BinaryWriter> serialize_int = (Int, writer) =>
            {
                writer.Write(Int);
            };

            Action<double, BinaryWriter> serialize_double = (dbl, writer) =>
            {
                writer.Write(dbl);
            };

            Action<string, BinaryWriter> serialize_string = (str, writer) =>
            {
                writer.Write(str);
            };

            Action<Dictionary<int, int>, BinaryWriter> serialize_dictionary_int_int = (str, writer) =>
            {
                WriteFile<int, int>(writer, str, serialize_int, serialize_int);
            };

            WriteFile<int, int>(binaryWriter, _filteredFeatures, serialize_int, serialize_int);
            WriteFile<int, double>(binaryWriter, _featureWeight, serialize_int, serialize_double);
            WriteFile<int, string>(binaryWriter, _featureById, serialize_int, serialize_string);
            WriteFile<string, Dictionary<int, int>>(binaryWriter, _bgrams, serialize_string, serialize_dictionary_int_int);
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

        #region N-Gramm

        short[] StringToByteArray(string word)
        {
            var data = System.Text.UnicodeEncoding.Unicode.GetBytes(word);
            short[] sdata = new short[data.Length / 2];
            Buffer.BlockCopy(data, 0, sdata, 0, data.Length);
            return sdata;
        }

        bool IsNGramValid(string bg)
        {
            int m = 0;

            for (int i = 0; i < _ngram; i++)
            {
                if (bg[i] == '?')
                    m++;
            }
            return m < 1;
        }

        List<string> MakeNGram(string word)
        {
            short[] b = StringToByteArray(word);
            short[] bg = new short[_ngram];
            List<string> ret = new List<string>();

            for (int i = -1; i < b.Length - (_ngram - 2); i++)
            {
                for (int g = i; g < i + _ngram; g++)
                {
                    bg[g - i] = (g < 0 || g >= b.Length) ? (byte)'$' : b[g];
                }

                byte[] result = new byte[_ngram * 2];
                Buffer.BlockCopy(bg, 0, result, 0, result.Length);
                var bgString = System.Text.UnicodeEncoding.Unicode.GetString(result);

                ret.Add(bgString);
            }
            return ret;
        }

        #endregion

        #region IFeatureManager

        public void Initialize(IEnumerable<Voucher> vouchers)
        {
            #region Split Words in Tri-Grams

            Dictionary<int, string> invWords = new Dictionary<int, string>();
            Dictionary<int, int> wordFrequency = new Dictionary<int, int>();
            Dictionary<string, Dictionary<int, int>> bgrams = new Dictionary<string, Dictionary<int, int>>();

            int iword = 0;

            Dictionary<string, int> exists = new Dictionary<string, int>();
            foreach (var tag in vouchers.SelectMany(x => x.OcrFeatures))
            {
                if (!exists.ContainsKey(tag))
                {
                    exists.Add(tag, iword);
                    invWords.Add(iword, tag);
                    wordFrequency.Add(iword, 1);

                    short[] b = StringToByteArray(tag);
                    short[] bg = new short[_ngram];

                    for (int i = -1; i < b.Length - (_ngram - 2); i++)
                    {
                        for (int g = i; g < i + _ngram; g++)
                        {
                            bg[g - i] = (g < 0 || g >= b.Length) ? (byte)'$' : b[g];
                        }

                        byte[] result = new byte[_ngram * 2];
                        Buffer.BlockCopy(bg, 0, result, 0, result.Length);
                        var bgString = System.Text.UnicodeEncoding.Unicode.GetString(result);

                        if (bgrams.ContainsKey(bgString))
                        {
                            if (bgrams[bgString].ContainsKey(iword))
                            {
                                bgrams[bgString][iword]++;
                            }
                            else
                            {
                                bgrams[bgString].Add(iword, 1);
                            }
                        }
                        else
                        {
                            bgrams.Add(bgString, new Dictionary<int, int> { { iword, 1 } });
                        }
                    }

                    iword++;
                }
                else
                {
                    wordFrequency[exists[tag]]++;
                }
            }
            exists.Clear();

            #endregion

            #region Compute similiarity matrix

            var sync = new Object();
            List<Tuple<int, int, double>> similiaritiesMerged = new List<Tuple<int, int, double>>();
            Action<List<Tuple<int, int, double>>> mergeSimiliarities = (s1) =>
                {
                    lock (sync)
                    {
                        similiaritiesMerged.AddRange(s1);
                    }
                };

            Parallel.ForEach<KeyValuePair<int, string>, List<Tuple<int, int, double>>>(invWords, // source collection
                                    () => new List<Tuple<int, int, double>>(), // method to initialize the local variable
                                    (words, loop, similiarities) => // method invoked by the loop on each iteration
                                    {
                                        Dictionary<int, double> matches = new Dictionary<int, double>();
                                        List<string> ngrams = MakeNGram(words.Value);
                                        foreach (var bg in ngrams.GroupBy(x => x))
                                        {
                                            if (IsNGramValid(bg.Key) && bgrams.ContainsKey(bg.Key))
                                            {
                                                foreach (var x in bgrams[bg.Key])
                                                {
                                                    if (matches.ContainsKey(x.Key))
                                                    {
                                                        matches[x.Key] += Math.Min(x.Value, bg.Count());
                                                    }
                                                    else
                                                    {
                                                        matches.Add(x.Key, Math.Min(x.Value, bg.Count()));
                                                    }
                                                }
                                            }
                                        }
                                        var ret = matches
                                            .Where(x => words.Key != x.Key)
                                            .Select(x => new Tuple<int, int, double>(words.Key, x.Key, x.Value / ((invWords[x.Key].Length - _ngram + 3) + ngrams.Count - x.Value)))
                                            .Where(x => x.Item3 >= _similiarityThreshold)
                                            .ToList();

                                        ret.ForEach(x => similiarities.Add(x));
                                        return similiarities; // value to be passed to next iteration
                                    },
                                    // Method to be executed when each partition has completed.
                                    // finalResult is the final value of subtotal for a particular partition.
                                    (finalResult) => mergeSimiliarities(finalResult)
                                    );




            #endregion

            #region Cluster words using Afinitty Propagation algorithm

            APClustering apCluster = new APClustering { Lam = 0.5, InvWords = invWords, Similiarities = similiaritiesMerged };
            apCluster.Cluster();

            invWords.ToList().ForEach(x =>
            {
                // check if data point
                if (!apCluster.Word2Cluster.ContainsKey(x.Key) && wordFrequency[x.Key] > 1)
                {
                    List<string> ngrams = MakeNGram(x.Value);
                    double match = ((double)ngrams.Where(w => IsNGramValid(w)).Count()) / x.Value.Length;
                    if (match >= 0.7)
                    {
                        apCluster.Word2Cluster.Add(x.Key, x.Key);
                    }
                }
            });


            #endregion

            #region Create Final Features Tri-Grams
            int ifeature = 0;
            Dictionary<string, int> existsFeature = new Dictionary<string, int>();

            foreach (var tag in apCluster.Word2Cluster.Values.Distinct().Select(x => invWords[x]))
            {
                if (!existsFeature.ContainsKey(tag))
                {
                    existsFeature.Add(tag, ifeature);

                    _featureById.Add(ifeature, tag);
                    _featureWeight.Add(ifeature, 1.0);

                    short[] b = StringToByteArray(tag);
                    short[] bg = new short[_ngram];

                    for (int i = -1; i < b.Length - (_ngram - 2); i++)
                    {
                        for (int g = i; g < i + _ngram; g++)
                        {
                            bg[g - i] = (g < 0 || g >= b.Length) ? (byte)'$' : b[g];
                        }

                        byte[] result = new byte[_ngram * 2];
                        Buffer.BlockCopy(bg, 0, result, 0, result.Length);
                        var bgString = System.Text.UnicodeEncoding.Unicode.GetString(result);

                        if (_bgrams.ContainsKey(bgString))
                        {
                            if (_bgrams[bgString].ContainsKey(ifeature))
                            {
                                _bgrams[bgString][ifeature]++;
                            }
                            else
                            {
                                _bgrams[bgString].Add(ifeature, 1);
                            }
                        }
                        else
                        {
                            _bgrams.Add(bgString, new Dictionary<int, int> { { ifeature, 1 } });
                        }
                    }

                    ifeature++;
                }
            }
            #endregion

            #region Compute feature importance/weights

            //Create Features to Vouchers Mappings
            Dictionary<int, List<Voucher>> feature2Vouchers = new Dictionary<int, List<Voucher>>();

            foreach (var voucher in vouchers)
            {
                List<Tuple<int, double>> features = ExpandFeatures(voucher.OcrFeatures.ToList());

                features.ForEach(x =>
                {
                    if (feature2Vouchers.ContainsKey(x.Item1))
                    {
                        feature2Vouchers[x.Item1].Add(voucher);
                    }
                    else
                    {
                        feature2Vouchers.Add(x.Item1, new List<Voucher> { voucher });
                    }
                });
            }

            // Compute feature importance/weights

            var totalDocsPerCateg = vouchers.GroupBy(x => x.TagName).ToDictionary(x => x.Key, x => x.Count());
            double totalCategs = totalDocsPerCateg.Count();
            Dictionary<Tuple<int, string>, double> weights = new Dictionary<Tuple<int, string>, double>();

            foreach (var feature in feature2Vouchers)
            {
                var categs = feature.Value.GroupBy(x => x.TagName);
                double noOfFeatureCategs = categs.Count();
                double f_idf = 1 - (noOfFeatureCategs - 1) / totalCategs;// (totalCategs != noOfFeatureCategs) ? 1 - 1 / Math.Log10(totalCategs / noOfFeatureCategs) : 0;
                _featureWeight[feature.Key] = Math.Exp(_weigthExpValue * (1 - f_idf));
            }

            //var tmp2 = _featureWeight.OrderBy(x => x.Value).Select(x => new Tuple<string, double>(_featureById[x.Key], x.Value)).ToArray();

            #endregion

            #region Filter features by Weights and create map to sequential Ids

            int index = 0;
            foreach (var featureId in _featureWeight.Where(x => x.Value >= 0.01))
            {
                _filteredFeatures.Add(featureId.Key, index++);
            }

            #endregion
        }


        public List<Tuple<int, double>> ReadWhenTrainingFeatures(List<string> ocrFeatures)
        {
            return ExpandFeatures(ocrFeatures).Select(x => new Tuple<int, double>(_filteredFeatures[x.Item1], x.Item2)).ToList();
        }

        public List<Tuple<int, double>> ReadWhenPredictingFeatures(List<string> ocrFeatures)
        {
            return ExpandFeatures(ocrFeatures).Select(x => new Tuple<int, double>(_filteredFeatures[x.Item1], x.Item2)).ToList();
        }

        public Voucher[] ReadUniqueVouchers(Voucher[] vouchers)
        {
            List<Voucher> ret = new List<Voucher>();
            Dictionary<VoucherFeatures, Voucher> unique = new Dictionary<VoucherFeatures, Voucher>();

            foreach (var voucher in vouchers)
            {
                var features = ReadWhenTrainingFeatures(voucher.OcrFeatures.ToList());
                var vFeatures = new VoucherFeatures { Features = features };

                if (features.Count > 0)
                {
                    if (unique.ContainsKey(vFeatures))
                    {
                        // check if vouchers are equal
                        var matchVoucher = unique[vFeatures];
                        if (!matchVoucher.TagName.Equals(voucher.TagName))
                        {
                            ret.Add(voucher);
                            // the same voucher identifiers multiple categories
                            //matchVoucher.TagName += "," + voucher.TagName;
                        }

                    }
                    else
                    {
                        ret.Add(voucher);
                        unique.Add(vFeatures, voucher);
                    }
                }
            }

            return ret.ToArray();
        }

        #endregion

        List<Tuple<int, double>> ExpandFeatures(List<string> ocrFeatures)
        {
            List<Tuple<int, double>> features = ocrFeatures.SelectMany(f =>
            {
                Dictionary<int, double> matches = new Dictionary<int, double>();
                List<string> ngrams = MakeNGram(f);
                foreach (var bg in ngrams.GroupBy(x => x))
                {
                    if (IsNGramValid(bg.Key) && _bgrams.ContainsKey(bg.Key))
                    {
                        foreach (var x in _bgrams[bg.Key])
                        {
                            if (matches.ContainsKey(x.Key))
                            {
                                matches[x.Key] += Math.Min(x.Value, bg.Count());
                            }
                            else
                            {
                                matches.Add(x.Key, Math.Min(x.Value, bg.Count()));
                            }
                        }
                    }
                }
                var ret = matches
                    .Select(x => new Tuple<int, double>(x.Key, x.Value / ((_featureById[x.Key].Length - _ngram + 3) + ngrams.Count - x.Value)))
                    .Where(x => x.Item2 >= _similiarityThreshold && _featureWeight[x.Item1] >= 0.01)
                    .GroupBy(x => x.Item1)
                    .Select(x => new Tuple<int, double>(x.Key, x.Max(y => y.Item2) /**  _featureWeight[x.Key]*/))
                    .ToList();

                return ret;

            }).Distinct().ToList();

            return features;
        }
    }
}
