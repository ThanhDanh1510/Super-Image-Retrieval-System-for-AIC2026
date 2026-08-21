// tools/data_prep/test_highlight_fix.js

function removeVietnameseDiacritics(str) {
  if (!str) return "";
  const normalized = str.normalize("NFC").toLowerCase();
  const accents = {
    '[àáảãạăằắẳẵặâầấẩẫậ]': 'a',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y',
    'đ': 'd'
  };
  let res = normalized;
  for (const [pattern, repl] of Object.entries(accents)) {
    res = res.replace(new RegExp(pattern, 'g'), repl);
  }
  return res;
}

function buildVietnameseCharRegex(term) {
  const charMap = {
    'a': '[aàáảãạăằắẳẵặâầấẩẫậAÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]',
    'd': '[dđDĐ]',
    'e': '[eèéẻẽẹêềếểễệEÈÉẺẼẸÊỀẾỂỄỆ]',
    'i': '[iìíỉĩịIÌÍỈĨỊ]',
    'o': '[oòóỏõọôồốổỗộơờớởỡợOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ]',
    'u': '[uùúủũụưừứửữựUÙÚỦŨỤƯỪỨỬỮỰ]',
    'y': '[yỳýỷỹỵYỲÝỶỸỴ]'
  };

  const rootTerm = removeVietnameseDiacritics(term);
  let pat = "";
  for (const ch of rootTerm) {
    if (charMap[ch]) {
      pat += charMap[ch];
    } else if (/\s/.test(ch)) {
      pat += '\\s+';
    } else {
      pat += ch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
  }
  return pat;
}

function buildVietnameseWordRegex(term) {
  const norm = term.normalize("NFC").trim();
  const unaccent = removeVietnameseDiacritics(norm);

  // Nếu từ có dấu (ví dụ: Khánh, Hòa, học, xe)
  if (norm !== unaccent) {
    let escaped = norm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    // Hỗ trợ cả 2 kiểu đặt dấu (hòa <-> hoà, thủy <-> thuỷ)
    escaped = escaped
      .replace(/([oôơ])([aăâáàảãạắằẳẵặấầẩẫậ])/gi, '($1$2|$2$1)')
      .replace(/([uư])([yýỳỷỹỵ])/gi, '($1$2|$2$1)');
    return escaped;
  } else {
    // Nếu từ không dấu -> hỗ trợ cả có dấu lẫn không dấu
    return buildVietnameseCharRegex(norm);
  }
}

const STOP_WORDS = new Set([
  "la", "va", "co", "cua", "o", "trong", "mot", "cac", "nhung", "cho", "voi", "duoc", "de", "thi", "ma", "ra", "vao",
  "là", "và", "có", "của", "ở", "trong", "một", "các", "những", "cho", "với", "được", "để", "thì", "mà", "ra", "vào",
  "the", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "a", "an", "with", "by", "as", "from"
]);

function extractSearchPatterns(query) {
  if (!query || typeof query !== "string" || !query.trim()) return [];
  
  const cleanQuery = query
    .normalize("NFC")
    .replace(/[.,?!:;()"'“”/…—\-_+*&^%$#@!~`\[\]{}\\\/]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();

  if (!cleanQuery) return [];

  const rawTerms = cleanQuery.split(/\s+/).filter(t => t.length >= 2);
  const matchedEnTerms = new Set();
  const matchedViTerms = new Set();

  if (cleanQuery.length >= 3) {
    matchedViTerms.add(cleanQuery);
  }

  rawTerms.forEach(term => {
    const unaccent = removeVietnameseDiacritics(term);
    if (STOP_WORDS.has(unaccent) || STOP_WORDS.has(term)) return;

    if (/^[a-zA-Z]+$/.test(term)) {
      matchedEnTerms.add(term);
    } else {
      matchedViTerms.add(term);
    }
  });

  const patterns = [];

  matchedEnTerms.forEach(enTerm => {
    if (enTerm.length < 2 || STOP_WORDS.has(enTerm.toLowerCase())) return;
    const escaped = enTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (enTerm.length >= 3) {
      patterns.push(`(?<=^|[^\\p{L}\\p{N}])(?:${escaped}(?:s|es|ed|ing|er|ers)?)(?=[^\\p{L}\\p{N}]|$)`);
    } else {
      patterns.push(`(?<=^|[^\\p{L}\\p{N}])${escaped}(?=[^\\p{L}\\p{N}]|$)`);
    }
  });

  matchedViTerms.forEach(viTerm => {
    if (viTerm.length < 2) return;
    const unaccent = removeVietnameseDiacritics(viTerm);
    if (STOP_WORDS.has(unaccent) || STOP_WORDS.has(viTerm)) return;
    const charPat = buildVietnameseWordRegex(viTerm);
    patterns.push(`(?<=^|[^\\p{L}\\p{N}])(?:${charPat})(?=[^\\p{L}\\p{N}]|$)`);
  });

  return Array.from(new Set(patterns));
}

function highlightMatchedText(fullText, query) {
  if (!fullText || fullText === "None" || fullText === "-") return fullText || "None";
  if (!query || typeof query !== "string" || !query.trim()) return fullText;

  const normalizedText = fullText.normalize("NFC");
  const patterns = extractSearchPatterns(query);
  if (patterns.length === 0) return normalizedText;

  patterns.sort((a, b) => b.length - a.length);

  try {
    const regex = new RegExp(`(${patterns.join("|")})`, "giu");
    return normalizedText.replace(regex, (match) => `<mark class="highlight-match">${match}</mark>`);
  } catch (err) {
    return normalizedText;
  }
}

const query = 'Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA. Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã thuộc tỉnh Khánh Hòa. Hỏi xã này có tên là gì? (tại thời điểm đó)';
const ocr = 'Xã Giang Ly, huyện Khánh Vĩnh tỉnh Khánh Hòa 04.08.2024 CÙNG EM';
const asr = '...tỉnh thành có thể kể đến như đắc nông đắc lắc lâm đồng bình phước tây ninh khánh hòa bình thuận ninh thuận...';

console.log('=== RESULT OCR ===\n', highlightMatchedText(ocr, query));
console.log('\n=== RESULT ASR ===\n', highlightMatchedText(asr, query));
