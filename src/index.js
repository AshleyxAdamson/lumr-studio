const PACE_ENUM = new Set(['natural', 'standard', 'fast', 'tight', 'hard', 'max', 'custom']);
const SOURCE_ENUM = new Set(['claude', 'pick', 'auto', 'creator']);
const KIND_ENUM = new Set(['pause', 'filler', 'stutter', 'repeat', 'false_start', 'off_topic', 'likes', 'other']);
const POS_ENUM = new Set(['noun', 'verb', 'adj', 'adv', 'pron', 'det', 'prep', 'conj', 'num', 'interj', 'filler', 'other']);
const PITCH_ENUM = new Set(['rising', 'falling', 'flat', 'unknown']);
const SENTENCE_POSITION_ENUM = new Set(['start', 'middle', 'end', 'whole']);
const ACTION_ENUM = new Set(['kept', 'put_back', 'rated_good', 'rated_bad', 'brought_back_words', 'cut_by_hand', 'kept_part']);
const FLAG_ENUM = new Set(['mid_sentence_out', 'mid_sentence_in', 'splice', 'removes_laugh', 'clips_beat', 're_entry', 'long_jump', 'fragment', 'tight']);
const FILLER_ENUM = new Set(['um', 'uh', 'er', 'ah', 'hmm', 'like', 'so', 'you know', 'i mean', 'basically', 'literally', 'right', 'actually', 'well']);


export function validateSend(obj) {
  // Root level validation
  if (!obj || typeof obj !== 'object') return 'Expected an object';
  if (obj.schema !== 1) return 'schema must be 1';
  if (typeof obj.send_id !== 'string') return 'send_id must be string';
  if (!obj.send_id.match(/^[A-Za-z0-9_-]{16,32}$/)) return 'send_id invalid format';
  if (typeof obj.plugin_version !== 'string') return 'plugin_version must be string';
  if (!obj.plugin_version.match(/^\d+\.\d+\.\d+$/)) return 'plugin_version invalid format';
  if (!Array.isArray(obj.records)) return 'records must be array';
  if (obj.records.length > 500) return 'records exceeds 500';

  // Check for unknown keys at root level
  const allowedRootKeys = new Set(['schema', 'send_id', 'plugin_version', 'records']);
  for (const key in obj) {
    if (!allowedRootKeys.has(key)) return `Unknown key at root: ${key}`;
  }

  // Validate each record
  for (let i = 0; i < obj.records.length; i++) {
    const rec = obj.records[i];
    const err = validateRecord(rec);
    if (err) return `Record ${i}: ${err}`;
  }

  return null;
}

function validateRecord(rec) {
  if (!rec || typeof rec !== 'object') return 'Record must be object';

  // Check for unknown keys
  const allowedKeys = new Set(['pace', 'proposed', 'context', 'creator', 'join']);
  for (const key in rec) {
    if (!allowedKeys.has(key)) return `Unknown key: ${key}`;
  }

  // pace
  if (!rec.pace || typeof rec.pace !== 'string') return 'pace must be string';
  if (!PACE_ENUM.has(rec.pace)) return `pace invalid: ${rec.pace}`;

  // proposed
  if (!rec.proposed || typeof rec.proposed !== 'object') return 'proposed must be object';
  const propErr = validateProposed(rec.proposed);
  if (propErr) return `proposed: ${propErr}`;

  // context
  if (!rec.context || typeof rec.context !== 'object') return 'context must be object';
  const ctxErr = validateContext(rec.context);
  if (ctxErr) return `context: ${ctxErr}`;

  // creator
  if (!rec.creator || typeof rec.creator !== 'object') return 'creator must be object';
  if (typeof rec.creator.action !== 'string') return 'creator.action must be string';
  if (!ACTION_ENUM.has(rec.creator.action)) return `creator.action invalid: ${rec.creator.action}`;
  const creatorKeys = new Set(['action']);
  for (const key in rec.creator) {
    if (!creatorKeys.has(key)) return `creator unknown key: ${key}`;
  }

  // join
  if (!rec.join || typeof rec.join !== 'object') return 'join must be object';
  const joinErr = validateJoin(rec.join);
  if (joinErr) return `join: ${joinErr}`;

  return null;
}

function validateProposed(prop) {
  const allowedKeys = new Set(['source', 'kind', 'length_s', 'words']);
  for (const key in prop) {
    if (!allowedKeys.has(key)) return `unknown key: ${key}`;
  }

  if (typeof prop.source !== 'string') return 'source must be string';
  if (!SOURCE_ENUM.has(prop.source)) return `source invalid: ${prop.source}`;

  if (typeof prop.kind !== 'string') return 'kind must be string';
  if (!KIND_ENUM.has(prop.kind)) return `kind invalid: ${prop.kind}`;

  if (typeof prop.length_s !== 'number') return 'length_s must be number';
  if (prop.length_s < 0 || prop.length_s > 3600) return 'length_s out of range';
  if (!isRoundedTo2Decimals(prop.length_s)) return 'length_s must be rounded to 2 decimals';

  if (typeof prop.words !== 'number') return 'words must be number';
  if (prop.words < 0 || prop.words > 10000) return 'words out of range';
  if (prop.words !== Math.floor(prop.words)) return 'words must be integer';

  return null;
}

function validateContext(ctx) {
  const allowedKeys = new Set(['before', 'after', 'sentence_position', 'laugh_within_s']);
  for (const key in ctx) {
    if (!allowedKeys.has(key)) return `unknown key: ${key}`;
  }

  if (!Array.isArray(ctx.before)) return 'before must be array';
  if (ctx.before.length > 3) return 'before exceeds 3';
  for (let i = 0; i < ctx.before.length; i++) {
    const err = validateWordShape(ctx.before[i]);
    if (err) return `before[${i}]: ${err}`;
  }

  if (!Array.isArray(ctx.after)) return 'after must be array';
  if (ctx.after.length > 3) return 'after exceeds 3';
  for (let i = 0; i < ctx.after.length; i++) {
    const err = validateWordShape(ctx.after[i]);
    if (err) return `after[${i}]: ${err}`;
  }

  if (typeof ctx.sentence_position !== 'string') return 'sentence_position must be string';
  if (!SENTENCE_POSITION_ENUM.has(ctx.sentence_position)) return `sentence_position invalid: ${ctx.sentence_position}`;

  if (ctx.laugh_within_s !== null && typeof ctx.laugh_within_s !== 'number') return 'laugh_within_s must be number or null';
  if (typeof ctx.laugh_within_s === 'number') {
    if (ctx.laugh_within_s < 0 || ctx.laugh_within_s > 600) return 'laugh_within_s out of range';
    if (!isRoundedTo2Decimals(ctx.laugh_within_s)) return 'laugh_within_s must be rounded to 2 decimals';
  }

  return null;
}

function validateWordShape(ws) {
  if (!ws || typeof ws !== 'object') return 'must be object';

  const allowedKeys = new Set(['pos', 'dur', 'gap_after', 'pitch', 'filler']);
  for (const key in ws) {
    if (!allowedKeys.has(key)) return `unknown key: ${key}`;
  }

  if (typeof ws.pos !== 'string') return 'pos must be string';
  if (!POS_ENUM.has(ws.pos)) return `pos invalid: ${ws.pos}`;

  if (typeof ws.dur !== 'number') return 'dur must be number';
  if (ws.dur < 0 || ws.dur > 10) return 'dur out of range';
  if (!isRoundedTo2Decimals(ws.dur)) return 'dur must be rounded to 2 decimals';

  if (typeof ws.gap_after !== 'number') return 'gap_after must be number';
  if (ws.gap_after < 0 || ws.gap_after > 60) return 'gap_after out of range';
  if (!isRoundedTo2Decimals(ws.gap_after)) return 'gap_after must be rounded to 2 decimals';

  if (typeof ws.pitch !== 'string') return 'pitch must be string';
  if (!PITCH_ENUM.has(ws.pitch)) return `pitch invalid: ${ws.pitch}`;

  if (ws.filler !== null && typeof ws.filler !== 'string') return 'filler must be string or null';
  if (typeof ws.filler === 'string') {
    if (!FILLER_ENUM.has(ws.filler)) return `filler invalid: ${ws.filler}`;
  }

  return null;
}

function validateJoin(join) {
  const allowedKeys = new Set(['gap_left_s', 'flags']);
  for (const key in join) {
    if (!allowedKeys.has(key)) return `unknown key: ${key}`;
  }

  if (join.gap_left_s !== null && typeof join.gap_left_s !== 'number') return 'gap_left_s must be number or null';
  if (typeof join.gap_left_s === 'number') {
    if (join.gap_left_s < 0 || join.gap_left_s > 60) return 'gap_left_s out of range';
    if (!isRoundedTo2Decimals(join.gap_left_s)) return 'gap_left_s must be rounded to 2 decimals';
  }

  if (!Array.isArray(join.flags)) return 'flags must be array';
  for (let i = 0; i < join.flags.length; i++) {
    const flag = join.flags[i];
    if (typeof flag !== 'string') return `flags[${i}] must be string`;
    if (!FLAG_ENUM.has(flag)) return `flags[${i}] invalid: ${flag}`;
  }

  return null;
}


function isRoundedTo2Decimals(num) {
  // Check if the number has at most 2 decimal places
  const rounded = Math.round(num * 100) / 100;
  return Math.abs(num - rounded) < 1e-9;
}

export function validateFeedback(obj) {
  if (!obj || typeof obj !== 'object') return 'Expected an object';
  if (obj.schema !== 1) return 'schema must be 1';
  if (typeof obj.feedback_id !== 'string') return 'feedback_id must be string';
  if (!obj.feedback_id.match(/^[A-Za-z0-9_-]{16,32}$/)) return 'feedback_id invalid format';
  if (typeof obj.plugin_version !== 'string') return 'plugin_version must be string';
  if (!obj.plugin_version.match(/^\d+\.\d+\.\d+$/)) return 'plugin_version invalid format';
  if (typeof obj.message !== 'string') return 'message must be string';
  if (obj.message.length < 1 || obj.message.length > 5000) return 'message length must be 1-5000';

  if (obj.name !== null && obj.name !== undefined) {
    if (typeof obj.name !== 'string') return 'name must be string or null';
    if (obj.name.length > 100) return 'name exceeds 100 characters';
  }

  if (obj.email !== null && obj.email !== undefined) {
    if (typeof obj.email !== 'string') return 'email must be string or null';
    if (obj.email.length > 200) return 'email exceeds 200 characters';
    if (obj.email.length > 0 && !obj.email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) return 'email invalid format';
  }

  // Check for unknown keys
  const allowedFeedbackKeys = new Set(['schema', 'feedback_id', 'plugin_version', 'message', 'name', 'email']);
  for (const key in obj) {
    if (!allowedFeedbackKeys.has(key)) return `Unknown key: ${key}`;
  }

  return null;
}

function getToday() {
  const now = new Date();
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, '0');
  const day = String(now.getUTCDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function checkAdminAuth(request, env) {
  if (!env.ADMIN_TOKEN) return false; // deletes stay locked until the owner sets ADMIN_TOKEN
  const authHeader = request.headers.get('Authorization');
  if (!authHeader) return false;
  const expected = `Bearer ${env.ADMIN_TOKEN}`;
  return authHeader === expected;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const method = request.method;

    // Rate limit check (except OPTIONS)
    if (method !== 'OPTIONS') {
      const ip = request.headers.get('cf-connecting-ip') || 'unknown';
      try {
        const { success } = await env.LIMITER.limit({ key: ip });
        if (!success) {
          return new Response(JSON.stringify({ error: 'Rate limit exceeded' }), {
            status: 429,
            headers: { 'Content-Type': 'application/json' }
          });
        }
      } catch (e) {
        // If rate limiter fails, let the request through
      }
    }

    if (method === 'OPTIONS') {
      return new Response(null, { status: 405, headers: { 'Access-Control-Allow-Origin': '*' } });
    }

    if (method === 'POST' && url.pathname === '/v1/sends') {
      let body;
      try {
        const raw = await request.text();
        if (new TextEncoder().encode(raw).length > 256 * 1024) {
          return new Response(JSON.stringify({ error: 'Body too large' }), {
            status: 413,
            headers: { 'Content-Type': 'application/json' }
          });
        }
        body = JSON.parse(raw);
      } catch (e) {
        return new Response(JSON.stringify({ error: 'Invalid JSON' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const validationError = validateSend(body);
      if (validationError) {
        return new Response(JSON.stringify({ error: validationError }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const today = getToday();
      const key = `sends/${today}/${body.send_id}.json`;

      // Check for duplicate (head returns null when the key is missing)
      if (await env.SENDS.head(key)) {
        return new Response(JSON.stringify({ error: 'Duplicate send_id' }), {
          status: 409,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Add received_at and store
      const toStore = { ...body, received_at: today };
      try {
        await env.SENDS.put(key, JSON.stringify(toStore));
      } catch (e) {
        return new Response(JSON.stringify({ error: 'Storage error' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      return new Response(JSON.stringify({ send_id: body.send_id, records: body.records.length }), {
        status: 201,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (method === 'DELETE' && url.pathname.startsWith('/v1/sends/')) {
      if (!checkAdminAuth(request, env)) {
        return new Response(JSON.stringify({ error: 'Unauthorized' }), {
          status: 401,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const sendId = url.pathname.slice('/v1/sends/'.length);
      if (!sendId) {
        return new Response(JSON.stringify({ error: 'send_id required' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Try to find and delete the send (it could be from any date)
      // For simplicity, we search recent dates
      let found = false;
      const now = new Date();
      for (let i = 0; i < 7; i++) {
        const d = new Date(now);
        d.setUTCDate(d.getUTCDate() - i);
        const year = d.getUTCFullYear();
        const month = String(d.getUTCMonth() + 1).padStart(2, '0');
        const day = String(d.getUTCDate()).padStart(2, '0');
        const dateStr = `${year}-${month}-${day}`;
        const key = `sends/${dateStr}/${sendId}.json`;

        if (await env.SENDS.head(key)) {
          await env.SENDS.delete(key);
          found = true;
          break;
        }
      }

      if (found) {
        return new Response(JSON.stringify({ deleted: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      } else {
        return new Response(JSON.stringify({ error: 'Not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    if (method === 'POST' && url.pathname === '/v1/feedback') {
      let body;
      try {
        body = await request.json();
      } catch (e) {
        return new Response(JSON.stringify({ error: 'Invalid JSON' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Check body size (32 KiB = 32768 bytes)
      const bodyJson = JSON.stringify(body);
      if (bodyJson.length > 32768) {
        return new Response(JSON.stringify({ error: 'Body too large' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const validationError = validateFeedback(body);
      if (validationError) {
        return new Response(JSON.stringify({ error: validationError }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const today = getToday();
      const key = `feedback/${today}/${body.feedback_id}.json`;

      // Check for duplicate (head returns null when the key is missing)
      if (await env.SENDS.head(key)) {
        return new Response(JSON.stringify({ error: 'Duplicate feedback_id' }), {
          status: 409,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Add received_at and store
      const toStore = { ...body, received_at: today };
      try {
        await env.SENDS.put(key, JSON.stringify(toStore));
      } catch (e) {
        return new Response(JSON.stringify({ error: 'Storage error' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      return new Response(JSON.stringify({ feedback_id: body.feedback_id }), {
        status: 201,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (method === 'DELETE' && url.pathname.startsWith('/v1/feedback/')) {
      if (!checkAdminAuth(request, env)) {
        return new Response(JSON.stringify({ error: 'Unauthorized' }), {
          status: 401,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const feedbackId = url.pathname.slice('/v1/feedback/'.length);
      if (!feedbackId) {
        return new Response(JSON.stringify({ error: 'feedback_id required' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // Try to find and delete the feedback (it could be from any date)
      let found = false;
      const now = new Date();
      for (let i = 0; i < 7; i++) {
        const d = new Date(now);
        d.setUTCDate(d.getUTCDate() - i);
        const year = d.getUTCFullYear();
        const month = String(d.getUTCMonth() + 1).padStart(2, '0');
        const day = String(d.getUTCDate()).padStart(2, '0');
        const dateStr = `${year}-${month}-${day}`;
        const key = `feedback/${dateStr}/${feedbackId}.json`;

        if (await env.SENDS.head(key)) {
          await env.SENDS.delete(key);
          found = true;
          break;
        }
      }

      if (found) {
        return new Response(JSON.stringify({ deleted: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      } else {
        return new Response(JSON.stringify({ error: 'Not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    return new Response(JSON.stringify({ error: 'Not found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
