using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace VoucherTagging
{
    public class Voucher : IEquatable<Voucher>, IComparable<Voucher>
    {
        public string TagName { get; set; }
        public int Id { get; set; }
        public int OrganizationId { get; set; }
        public string Company { get; set; }
        public string OcrPath { get; set; }
        public List<string> OcrFeatures { get; set; }

        #region IEquatable

        public bool Equals(Voucher other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Id == other.Id;
        }

        #endregion

        #region IComparable

        public int CompareTo(Voucher other)
        {
            return this.Id > other.Id ? 1 : (this.Id == other.Id ? 0 : -1);
        }

        #endregion

        #region object

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            return obj is Voucher && Equals((Voucher)obj);
        }

        public override int GetHashCode()
        {
            return Id;
        }

        public static bool operator ==(Voucher left, Voucher right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(Voucher left, Voucher right)
        {
            return !Equals(left, right);
        }

        public override string ToString()
        {
            return Id.ToString(CultureInfo.InvariantCulture);
        }

        #endregion

        #region For test

        static public void WriteVouchers(Voucher[] vouchers, string file)
        {
            using (FileStream fs = File.OpenWrite(file))
            {
                using (BinaryWriter writer = new BinaryWriter(fs))
                {
                    writer.Write(vouchers.Length);
                    foreach (var voucher in vouchers)
                    {
                        // write attributes
                        writer.Write(voucher.TagName);
                        writer.Write(voucher.Id);
                        writer.Write(voucher.OrganizationId);
                        writer.Write(voucher.Company);
                        writer.Write(voucher.OcrPath);

                        // write OCR features
                        writer.Write(voucher.OcrFeatures.Count);
                        foreach (var f in voucher.OcrFeatures)
                        {
                            writer.Write(f);
                        }
                    }
                }
            }
        }

        static public Voucher[] ReadVouchers(string file)
        {
            Voucher[] result;
            using (FileStream fs = File.OpenRead(file))
            {
                using (BinaryReader reader = new BinaryReader(fs))
                {
                    int count = reader.ReadInt32();
                    result = new Voucher[count];

                    for (int i = 0; i < count; i++)
                    {
                        var voucher = new Voucher
                        {
                            TagName = reader.ReadString(),
                            Id = reader.ReadInt32(),
                            OrganizationId = reader.ReadInt32(),
                            Company = reader.ReadString(),
                            OcrPath = reader.ReadString()
                        };

                        // read OCR features
                        int nOcr = reader.ReadInt32();
                        voucher.OcrFeatures = new List<string>();
                        for (int y = 0; y < nOcr; y++)
                        {
                            voucher.OcrFeatures.Add(reader.ReadString());
                        }
                        result[i] = voucher;
                    }
                }
            }
            return result;
        }

        static public Voucher[] LoadVouchers(string[] paths)
        {
            Dictionary<int, Voucher> vouchers = new Dictionary<int, Voucher>();
            {
                TextFieldParser parser = new TextFieldParser(paths[0]);
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                bool first = true;
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    if (first)
                    {
                        first = false;
                        continue;
                    }

                    int Id = int.Parse(fields[0]);
                    int OId = int.Parse(fields[1]);
                    string Company = fields[5];
                    string OCR = fields[7];
                    vouchers.Add(Id, new Voucher { Id = Id, OrganizationId = OId, Company = Company, OcrPath = OCR });
                }
                parser.Close();
            }


            {
                TextFieldParser parser = new TextFieldParser(paths[1]);
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");
                bool first = true;
                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();
                    if (first)
                    {
                        first = false;
                        continue;
                    }

                    int Id = int.Parse(fields[0]);
                    string Tag = fields[4];
                    vouchers[Id].TagName = Tag;
                }

                parser.Close();
            }

            foreach (var voucher in vouchers)
            {
                List<byte> characters = new List<byte>();
                string ocr_path = string.Format(@"{0}\{1}-original.csv", paths[2], voucher.Value.OcrPath);
                voucher.Value.OcrFeatures = LoadOCRData(ocr_path);
            }
            return vouchers.Values.ToArray();
        }

        static public List<string> LoadOCRData(string path)
        {
            List<OcrInput> ocrInput = new List<OcrInput>();
            TextFieldParser parser = new TextFieldParser(path);
            parser.TextFieldType = FieldType.Delimited;
            parser.SetDelimiters(",");
            bool first = true;
            while (!parser.EndOfData)
            {
                string[] fields = parser.ReadFields();
                if (first)
                {
                    first = false;
                    continue;
                }

                var ocr = new OcrInput { Character = byte.Parse(fields[0]), Confidence = double.Parse(fields[1]) };
                if (ocr.Confidence < 0 || ocr.Confidence > 1)
                {
                    throw new ArgumentOutOfRangeException();
                }
                ocrInput.Add(ocr);
            }
            parser.Close();
            return LoadOCRData(ocrInput);
        }

        #endregion

        static public List<string> LoadOCRData(List<OcrInput> ocrInput, double confidence = 0.8)
        {
            HashSet<byte> replace = new HashSet<byte> { (byte)'=', (byte)'(', (byte)')', (byte)'-', (byte)'[', (byte)']', (byte)'.', (byte)';', (byte)':', (byte)'#' };
            List<byte> characters = new List<byte>();
            foreach (var input in ocrInput)
            {
                var character = input.Character;

                if (character == 13 || character == 32 || character == (byte)'.' || character == (byte)',' ||
                    character == (byte)';' || character == (byte)':')
                {
                    characters.Add((byte)'\r');
                }
                else if (replace.Contains(character) || Char.IsDigit((char)character))
                {
                    continue;
                }
                else if (input.Confidence < confidence)
                {
                    characters.Add((byte)'?');
                }
                else
                {
                    characters.Add((byte)character);
                }
            }

            var matchOcr = System.Text.ASCIIEncoding.Default.GetString(characters.ToArray());
            var ocr = matchOcr.Split(new char[] { '\r', '/', '\\' }).Where(x => Regex.IsMatch(x, "[a-zA-Z]") && x.Length > 3 && x.Length < 20).Select(x => x.ToLowerInvariant()).ToList();
            return ocr;
        }

    }
}
