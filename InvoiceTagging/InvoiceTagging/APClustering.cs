using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VoucherTagging
{
    /// <summary>
    /// Afinity propagation clustering used to cluster similiar words
    /// Each cluster is represented by so called "exempler" word. 
    /// </summary>
    internal class APClustering
    {
        /// <summary>
        /// Mapping each word in vouchers to unique id. When refer to the word (feature), we use most of the time this Id.
        /// </summary>
        public Dictionary<int, string> InvWords { get; set; }

        /// <summary>
        /// User friendly cluster representation
        /// </summary>
        public Dictionary<string, List<string>> Exemplers { get; private set; }

        /// <summary>
        /// Learning rate
        /// </summary>
        public double Lam { get; set; }

        /// <summary>
        /// Maps each vouchers' word to the cluster it belongs. 
        /// </summary>
        public Dictionary<int, int> Word2Cluster { get; private set; }

        /// <summary>
        /// Word to word similiarity using n-grams Jaccard distance
        /// </summary>
        public List<Tuple<int, int, double>> Similiarities;

        /// <summary>
        /// Similiarity matrix
        /// </summary>
        Dictionary<int, Dictionary<int, double>> S;

        /// <summary>
        /// Responsibility matrix
        /// </summary>
        Dictionary<int, Dictionary<int, double>> R;

        /// <summary>
        /// Availability matrix
        /// </summary>
        Dictionary<int, Dictionary<int, double>> A;

        public APClustering()
        {
            Word2Cluster = new Dictionary<int, int>();
        }

        public void Cluster()
        {
            var s1 = Similiarities
                .GroupBy(x => x.Item1)
                .ToDictionary(x => x.Key, x => x.ToDictionary(y => y.Item2, y => /*0.0001 * rand.NextDouble() +*/ (y.Item3 - 1.0)));

            S = s1;
            R = s1.ToDictionary(x => x.Key, x => x.Value.ToDictionary(y => y.Key, y => 0.0));
            A = s1.ToDictionary(x => x.Key, x => x.Value.ToDictionary(y => y.Key, y => 0.0));

            double s_k_k = -0.6;// _similiarityThreshold - 1.1;// similiarities.Min(x => x.Item3 - 1.0);
            int noOfExemplers = int.MaxValue;

            foreach (var i in S)
            {
                R[i.Key].Add(i.Key, 0.0);
                A[i.Key].Add(i.Key, 0.0);
            }

            for (int it = 0; it < 100; it++)
            {
                // ================= Responsibilities================================== // 
                // self responsibilities
                // R(k,k) = S(k,k) - max(S(k,k')), then i = k
                foreach (var i in S)
                {
                    double max_s_i_kk = i.Value.Count() > 1 ? i.Value.Max(x => x.Value) : 0.0;
                    R[i.Key][i.Key] = Lam * (s_k_k - max_s_i_kk) + (1 - Lam) * R[i.Key][i.Key];
                }

                // R(i,k) = S(i,k) - max(S(i,k') + a(i,k'))
                foreach (var i in S)
                {
                    double then_i_equal_i = s_k_k + R[i.Key][i.Key];
                    foreach (var k in i.Value)
                    {
                        double s_i_k = k.Value;
                        var neighbours = i.Value.Where(x => x.Key != k.Key);
                        double max_s_i_kk = Math.Max(then_i_equal_i, (neighbours.Count() > 0) ? neighbours.Select(x => A[i.Key][x.Key] + x.Value).Max() : double.MinValue);
                        R[i.Key][k.Key] = (Lam) * (s_i_k - max_s_i_kk) + (1.0 - Lam) * R[i.Key][k.Key];
                    }
                }

                // ================= Availabilities================================== // 
                foreach (var i in S)
                {
                    // self availability
                    //A(i,k) = SUM( MAX{0,R(ii != k,k)} ), Then i == k, ii != k
                    double a_k_k = S[i.Key].Select(x => Math.Max(0.0, R[x.Key][i.Key])).Sum();
                    A[i.Key][i.Key] = Lam * a_k_k + (1.0 - Lam) * A[i.Key][i.Key];

                    foreach (var k in i.Value)
                    {
                        //A(i,k) = MIN{0,  [ R(k,k) + SUM( MAX{0, r(ii,k)})] } , Then i != k  

                        // 1(i)->2(k)<-3(i)
                        // R(1,2) + R(3,2)
                        // k'th neighbours
                        double r_ii_k = S[k.Key].Where(x => x.Key != i.Key).Select(x => Math.Max(0.0, R[x.Key][k.Key])).Sum();
                        double a_i_k = Math.Min(0, R[k.Key][k.Key] + r_ii_k);
                        A[i.Key][k.Key] = Lam * a_i_k + (1.0 - Lam) * A[i.Key][k.Key];
                    }
                }

                // choose exemplers
                // For data point i, the value of k (data point) that maximizes a(i,k)+ r(i,k) either identifies data point i as an exemplar if k = i, 
                // or identifies the data point that is the exemplar for data point i. 
                Dictionary<int, List<int>> exemplers = new Dictionary<int, List<int>>();
                Dictionary<int, int> dataPoint = new Dictionary<int, int>();

                foreach (var i in S)
                {
                    int exempler = i.Key;
                    double max = A[i.Key][i.Key] + R[i.Key][i.Key];

                    foreach (var k in i.Value)
                    {
                        double tmp = A[i.Key][k.Key] + R[i.Key][k.Key];
                        if (tmp > max && !dataPoint.ContainsKey(k.Key)) // mus not be already allocated to data point by previous iteration(s)
                        {
                            exempler = k.Key;
                        }
                    }

                    // save new exempler
                    if (!exemplers.ContainsKey(exempler))
                    {
                        exemplers.Add(exempler, new List<int>());
                    }

                    // make data points list
                    if (exempler != i.Key && !exemplers.ContainsKey(i.Key))
                    {
                        // neighbour chooses as example, so i'th point becoems data point
                        dataPoint.Add(i.Key, exempler);
                        exemplers[exempler].Add(i.Key);
                    }
                }

                if (exemplers.Count() < noOfExemplers)
                {
                    Exemplers = exemplers.ToDictionary(x => InvWords[x.Key], x => x.Value.Select(y => InvWords[y]).ToList());
                    noOfExemplers = exemplers.Count;
                }
                else
                {

                    InvWords.ToList().ForEach(x =>
                    {
                        // check if data point
                        if (dataPoint.ContainsKey(x.Key))
                        {
                            Word2Cluster.Add(x.Key, dataPoint[x.Key]);
                        }
                        else if (exemplers.ContainsKey(x.Key))
                        {
                            Word2Cluster.Add(x.Key, x.Key);
                        }
                    });
                    break;
                }
            }
        }
    }
}
