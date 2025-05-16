#include "catch2/catch_all.hpp"

#include <string>
#include <vector>

#include "lingodb/runtime/StringRuntime.h"

#include <iostream>
using namespace lingodb::runtime;

namespace {
static void cleanUp(VarLen32& string) {
   if (!string.isShort()) {
      const auto* ptr = string.getPtr();
      delete[] ptr;
   }
}
} // namespace

// Testing Length -----------------------------------------------------------------------------------------------

// -- single byte
TEST_CASE("Length:SingleByte") {
   std::string characters[] = {
      "H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d",
      "D", "O", "N", "'", "T", "P", "A", "N", "I", "C",
      "T", "i", "m", "e", "i", "s", "a", "n", "i", "l", "l", "u", "s", "i", "o", "n", ".", "L", "u", "n", "c", "h", "t", "i", "m", "e", "d", "o", "u", "b", "l", "y", "s", "o", "."};

   for (std::size_t start = 0; start < std::size(characters); start++) {
      std::string testString = "";

      for (std::size_t current = start; current < std::size(characters); current++) {
         testString += characters[current];
         VarLen32 str = VarLen32::fromString(testString);
         REQUIRE(StringRuntime::len(str) == (int64_t) (current - start + 1));
         cleanUp(str);
      }
   }
}

// -- multibyte characters lengths: 2, 3, 4

TEST_CASE("Length:MultiByte") {
   /*
        Randomly generated strings and lengths.
        At least one length below 12
    */

   VarLen32 str;

   // 2 Bytes
   str = VarLen32::fromString("\u0093\u0722\u0151\u0559\u05F9\u048B\u07ED\u04AC");
   REQUIRE(StringRuntime::len(str) == 8);
   cleanUp(str);

   str = VarLen32::fromString("\u063A\u052C\u01EE\u01CB\u06A3\u03C7\u01B0\u062E\u0481\u01E7\u01A9\u0731\u023E");
   REQUIRE(StringRuntime::len(str) == 13);
   cleanUp(str);

   str = VarLen32::fromString("\u051F\u00F6\u03B8\u015E\u0313\u01A3\u06DE\u044D\u06E4\u0553\u0629\u05AA\u053F\u042C\u0267\u0772\u069E\u00AA\u0458\u032C\u056B");
   REQUIRE(StringRuntime::len(str) == 21);
   cleanUp(str);

   // 3 Bytes
   str = VarLen32::fromString("\u1D02\u8B98\uC239\u2ADF\uADEB\u6E12\u2030\u5925\u3717\uD203\u9FBD\u3047\u6119");
   REQUIRE(StringRuntime::len(str) == 13);
   cleanUp(str);

   str = VarLen32::fromString("\uBFBC\u2F68\u0BA3\u4072\u6692\uAC06\u88D3\u152E\uD375\u5DE3\uB473\uF78E\uF648\u8BD2\u8FAA\u1B91\uAFF4\u4093\u9D41\u7F3F\u8B20\uAC75");
   REQUIRE(StringRuntime::len(str) == 22);
   cleanUp(str);

   str = VarLen32::fromString("\u272A\uA88F\uEF5C\u999D\uA6E0\u71BF\u55BC\uC600\uFE61");
   REQUIRE(StringRuntime::len(str) == 9);
   cleanUp(str);

   // 4 Bytes
   str = VarLen32::fromString("\U0009EF2B\U0008BDFB\U00045E9F\U000680DC\U00034DE0\U000BD75A\U00030A47\U00020378\U0002D572\U0004AFC8\U000106ED\U000492A9\U0010968E\U00032A0F\U000D8399\U000134C3\U000495A0\U0009DBBD\U0001B305\U000DF024");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);

   str = VarLen32::fromString("\U0001325A\U000F6B32\U000F7013");
   REQUIRE(StringRuntime::len(str) == 3);
   cleanUp(str);

   str = VarLen32::fromString("\U000593FD\U00062603\U000A921E\U0001637A\U0004E61A\U000189D5\U000895C1\U000CC85F\U000A8EBD\U0006E79F\U000332D8\U0002817D\U000B2E3A\U00073307\U0006BC55\U000EBBA9\U00069727\U0004213E\U0008BF9E\U000A6453\U000ACDAB\U0006C051\U0001CC8F\U000980E3\U000EA1EB");
   REQUIRE(StringRuntime::len(str) == 25);
   cleanUp(str);

   // Mixture

   str = VarLen32::fromString("\u16AC\u06C2\U0008BE37\u0237\u0138\u0044\U000C4401\u0024\u0062\u0146\u9DA4\u004C\U00076719\u01F2\u05AE\u0267\u207B\u0077\u0051\U00068704");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);

   str = VarLen32::fromString("\u0027\uC638\U000A87DE\uE8D5\u17AA\uC529\U00032F8E\U0002D7FC\U00036186\u45AA\u0033\u0442\u005E\u0044\uAC34\u0040\U0002A1BE\u6113\uA272\u0169");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);

   str = VarLen32::fromString("\uF9EE\u05C3\u6120\u0044\u6EFF\u0020\u5929\uA655\u02AA\u1FC7\u0025\u0397\u007E\u1D6C\U000F5E22\u0696\u0068\u063D\u3FF1\U0004E934");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);

   str = VarLen32::fromString("\u0047\u0052\U000EA560\u003F\u0035\u004B\u01ED\u07E7\u02F2\u032E\uA8AC\uCE1D\u30CE\uABC5\u00F7\u005D\U000C3421\u003F\U0004AB27\u0205");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);

   str = VarLen32::fromString("\u005F\uF2FE\U0002FBD8\u03B3\u01EF\u0588\u0075\u003A\u0033\u005C\uC45A\u7FA0\u059A\u07CD\u045C\u0648\u0053\uB00C\u017E\U00018707");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);

   str = VarLen32::fromString("\u04D0\U000FDE46\U000B0A45\u0057\u0727\u022B\uA3BC\u0020\uA871\u003C\u0430\u0031\u038B\U00066A4F\u166B\u0159\u424F\U00039FC9\u0073\u068D");
   REQUIRE(StringRuntime::len(str) == 20);
   cleanUp(str);
}

// -- Edge Unicode points
TEST_CASE("Length:EdgePoints") {
   std::string edgePoints[] = {
      std::string(1, '\0'),
      "\u007F",
      "\u0080",
      "\u07FF",
      "\u0800",
      "\uFFFF",
      "\U00010000",
      "\U0010FFFF"};

   struct TestCase {
      int length;
      std::string testString;
   };

   std::vector<TestCase> testCases = {
      {0, ""}};

   for (std::string point : edgePoints) {
      int length = testCases.size();
      for (int i = 0; i < length; i++) {
         TestCase testCase = testCases[i];
         testCases.push_back({testCase.length + 1,
                              testCase.testString + point});
      }
   }

   for (TestCase testCase : testCases) {
      VarLen32 str = VarLen32::fromString(testCase.testString);
      REQUIRE(StringRuntime::len(str) == testCase.length);
      cleanUp(str);
   }
}

// Testing Substring -----------------------------------------------------------------------------------------------
namespace {

void testSubstrWithCleanup(const VarLen32& testString, const std::string& expected, int64_t from, int64_t len) {
   VarLen32 exp = VarLen32::fromString(expected);
   VarLen32 res = StringRuntime::substr(testString, from, len);
   REQUIRE(StringRuntime::compareEq(res, exp));
   cleanUp(exp);
   cleanUp(res);
}

void testSubstringFromChar(std::string characters[], size_t length) {
   std::string testString = "";

   /*
    * position in the array: starting character;
    * elements in the vector: substrings starting with said character
   */
   std::vector<std::vector<VarLen32>> subStrings(length);

   for (std::size_t start = 0; start < length; start++) {
      testString += characters[start];
      std::string subString = "";
      for (std::size_t current = start; current < length; current++) {
         subString += characters[current];
         VarLen32 str = VarLen32::fromString(subString);
         subStrings[start].push_back(str);
      }
   }

   VarLen32 str = VarLen32::fromString(testString);

   for (std::size_t startingCharacter = 0; startingCharacter < length; startingCharacter++) {
      for (std::size_t end = 0; end < subStrings[startingCharacter].size(); end++) {
         VarLen32& subString = subStrings[startingCharacter][end];
         VarLen32 res = StringRuntime::substr(str, startingCharacter + 1, end + 1);
         REQUIRE(StringRuntime::compareEq(
            res,
            subString));
         cleanUp(res);
      }
   }

   cleanUp(str);
   for (auto& list : subStrings) {
      for (auto& elem : list) {
         cleanUp(elem);
      }
   }
}
} // namespace

// -- single byte
TEST_CASE("Substr:SingleByte") {
   // TODO convert to std::array
   std::string characters[] = {
      "H", "e", "l", "l", "o", " ", "W", "o", "r", "l", "d",
      "D", "O", "N", "'", "T", "P", "A", "N", "I", "C",
      "T", "i", "m", "e", "i", "s", "a", "n", "i", "l", "l", "u", "s", "i", "o", "n", ".", "L", "u", "n", "c", "h", "t", "i", "m", "e", "d", "o", "u", "b", "l", "y", "s", "o", "."};

   testSubstringFromChar(characters, std::size(characters));
}

// -- multibyte characters lengths: 2, 3, 4

TEST_CASE("Substr:MultiByte") {
   std::string characters[] = {
      "\u0093", "\u0722", "\u0151", "\u0559", "\u05f9", "\u048b", "\u07ed", "\u04ac", "\u063a", "\u052c", "\u01ee", "\u01cb",
      "\u06a3", "\u03c7", "\u01b0", "\u062e", "\u0481", "\u01e7", "\u01a9", "\u0731", "\u023e", "\u051f", "\u00f6", "\u03b8",
      "\u015e", "\u0313", "\u01a3", "\u06de", "\u044d", "\u06e4", "\u0553", "\u0629", "\u05aa", "\u053f", "\u042c", "\u0267",
      "\u0772", "\u069e", "\u00aa", "\u0458", "\u032c", "\u056b", "\u1d02", "\u8b98", "\uc239", "\u2adf", "\uadeb", "\u6e12",
      "\u2030", "\u5925", "\u3717", "\ud203", "\u9fbd", "\u3047", "\u6119", "\ubfbc", "\u2f68", "\u0ba3", "\u4072", "\u6692",
      "\uac06", "\u88d3", "\u152e", "\ud375", "\u5de3", "\ub473", "\uf78e", "\uf648", "\u8bd2", "\u8faa", "\u1b91", "\uaff4",
      "\u4093", "\u9d41", "\u7f3f", "\u8b20", "\uac75", "\u272a", "\ua88f", "\uef5c", "\u999d", "\ua6e0", "\u71bf", "\u55bc",
      "\uc600", "\ufe61", "\U0009EF2B", "\U0008BDFB", "\U00045E9F", "\U000680DC", "\U00034DE0", "\U000BD75A", "\U00030A47", "\U00020378",
      "\U0002D572", "\U0004AFC8", "\U000106ED", "\U000492A9", "\U0010968E", "\U00032A0F", "\U000D8399", "\U000134C3", "\U000495A0",
      "\U0009DBBD", "\U0001B305", "\U000DF024", "\U0001325A", "\U000F6B32", "\U000F7013", "\U000593FD", "\U00062603", "\U000A921E",
      "\U0001637A", "\U0004E61A", "\U000189D5", "\U000895C1", "\U000CC85F", "\U000A8EBD", "\U0006E79F", "\U000332D8", "\U0002817D",
      "\U000B2E3A", "\U00073307", "\U0006BC55", "\U000EBBA9", "\U00069727", "\U0004213E", "\U0008BF9E", "\U000A6453", "\U000ACDAB",
      "\U0006C051", "\U0001CC8F", "\U000980E3", "\U000EA1EB", "\u16AC", "\u06C2", "\U0008BE37", "\u0237", "\u0138", "\u0044",
      "\U000C4401", "\u0024", "\u0062", "\u0146", "\u9DA4", "\u004C", "\U00076719", "\u01F2", "\u05AE", "\u0267", "\u207B",
      "\u0077", "\u0051", "\U00068704", "\u0027", "\uC638", "\U000A87DE", "\uE8D5", "\u17AA", "\uC529", "\U00032F8E", "\U0002D7FC",
      "\U00036186", "\u45AA", "\u0033", "\u0442", "\u005E", "\u0044", "\uAC34", "\u0040", "\U0002A1BE", "\u6113", "\uA272",
      "\u0169", "\uF9EE", "\u05C3", "\u6120", "\u0044", "\u6EFF", "\u0020", "\u5929", "\uA655", "\u02AA", "\u1FC7", "\u0025",
      "\u0397", "\u007E", "\u1D6C", "\U000F5E22", "\u0696", "\u0068", "\u063D", "\u3FF1", "\U0004E934", "\u0047", "\u0052",
      "\U000EA560", "\u003F", "\u0035", "\u004B", "\u01ED", "\u07E7", "\u02F2", "\u032E", "\uA8AC", "\uCE1D", "\u30CE",
      "\uABC5", "\u00F7", "\u005D", "\U000C3421", "\u003F", "\U0004AB27", "\u0205", "\u005F", "\uF2FE", "\U0002FBD8", "\u03B3",
      "\u01EF", "\u0588", "\u0075", "\u003A", "\u0033", "\u005C", "\uC45A", "\u7FA0", "\u059A", "\u07CD", "\u045C", "\u0648",
      "\u0053", "\uB00C", "\u017E", "\U00018707", "\u04D0", "\U000FDE46", "\U000B0A45", "\u0057", "\u0727", "\u022B", "\uA3BC",
      "\u0020", "\uA871", "\u003C", "\u0430", "\u0031", "\u038B", "\U00066A4F", "\u166B", "\u0159", "\u424F", "\U00039FC9",
      "\u0073", "\u068D"};

   testSubstringFromChar(characters, std::size(characters));
}

TEST_CASE("Substr:OutOfBounds") {
   VarLen32 testString = VarLen32::fromString("The Answer to the Great Question... Of Life, the Universe and Everything... Is... Forty-two");

   std::string testCase = "Answer to ";
   testSubstrWithCleanup(testString, testCase, 5, 10);

   testCase = "The Answer to";
   testSubstrWithCleanup(testString, testCase, -1, 15);
   testSubstrWithCleanup(testString, testCase, -10, 24);

   testCase = "The Answer to ";
   testSubstrWithCleanup(testString, testCase, 0, 15);

   testCase = "The Answer to the";
   testSubstrWithCleanup(testString, testCase, 1, 17);

   testCase = "o";
   testSubstrWithCleanup(testString, testCase, 91, 1);

   testCase = testString;
   testSubstrWithCleanup(testString, testCase, 1, 100);
   testSubstrWithCleanup(testString, testCase, 1, 91);

   testCase = "";
   testSubstrWithCleanup(testString, testCase, -10, 5);
   testSubstrWithCleanup(testString, testCase, 10, -1);
   testSubstrWithCleanup(testString, testCase, 1, 0);
   testSubstrWithCleanup(testString, testCase, 91, 0);
   testSubstrWithCleanup(testString, testCase, 92, 0);
   testSubstrWithCleanup(testString, testCase, 92, 1);
   testSubstrWithCleanup(testString, testCase, 100, 0);
   testSubstrWithCleanup(testString, testCase, 100, 10);

   testCase = "he Answer to the Great Question... Of Life, the Universe and Everything... Is... Forty-tw";
   testSubstrWithCleanup(testString, testCase, 2, 89);

   cleanUp(testString);
}

// Testing Like -----------------------------------------------------------------------------------------------

TEST_CASE("Like:Simple") {
   VarLen32 testLike = VarLen32::fromString("Jos_!");

   VarLen32 testCandidate = VarLen32::fromString("Jose!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("José!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   cleanUp(testLike);
}

TEST_CASE("Like:Combinations") {
   VarLen32 testLike;
   VarLen32 testCandidate;

   // Multi-byte Wildcard
   testLike = VarLen32::fromString("J_sé!");

   testCandidate = VarLen32::fromString("José!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // 'o' matches '_'
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jéssé!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == false); // two chars after J
   cleanUp(testCandidate);

   cleanUp(testCandidate);

   // % Matching
   testLike = VarLen32::fromString("J_sé!");

   testCandidate = VarLen32::fromString("José!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // 'o' matches '_'
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jéssé!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == false); // two chars after J
   cleanUp(testCandidate);

   cleanUp(testLike);

   // Matching across boundaries

   testLike = VarLen32::fromString("J%!");

   testCandidate = VarLen32::fromString("José!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jośę!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // two multi-byte chars
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("J!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // zero or more
   cleanUp(testCandidate);

   cleanUp(testLike);

   // Combining Characters

   testLike = VarLen32::fromString("Jo_%!");

   testCandidate = VarLen32::fromString("Joé!"); // composed form
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jo\u0065\u0301!"); // decomposed é (e + ́)
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // if normalized
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Joe!"); // ASCII e
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // match either form
   cleanUp(testCandidate);

   cleanUp(testLike);

   // UTF-8 Boundary with %

   testLike = VarLen32::fromString("%é!");

   testCandidate = VarLen32::fromString("José!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jøse!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == false);
   cleanUp(testCandidate);

   cleanUp(testLike);

   // Multiple _

   testLike = VarLen32::fromString("____!");

   testCandidate = VarLen32::fromString("José!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // 4 codepoints
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jose!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // ASCII only
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jósé!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true); // 5 codepoints
   cleanUp(testCandidate);

   cleanUp(testLike);

   // Start and End Anchors

   testLike = VarLen32::fromString("%é");

   testCandidate = VarLen32::fromString("Café");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Cafe");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == false);
   cleanUp(testCandidate);

   cleanUp(testLike);

   // Non-Match Edge Cases

   testLike = VarLen32::fromString("Jo%!");

   testCandidate = VarLen32::fromString("Jóse"); // no '!'
   REQUIRE(StringRuntime::like(testCandidate, testLike) == false);
   cleanUp(testCandidate);

   testCandidate = VarLen32::fromString("Jośé!!");
   REQUIRE(StringRuntime::like(testCandidate, testLike) == true);
   cleanUp(testCandidate);

   cleanUp(testLike);
}
