import { test } from 'node:test';
import assert from 'node:assert';
import { validateSend, validateFeedback } from '../src/index.js';

const validSend = {
  schema: 1,
  send_id: 'aBcDeFgHiJkLmNoPqRsT',
  plugin_version: '0.2.0',
  records: [
    {
      pace: 'natural',
      proposed: {
        source: 'claude',
        kind: 'pause',
        length_s: 1.23,
        words: 5
      },
      context: {
        before: [
          { pos: 'noun', dur: 0.5, gap_after: 0.1, pitch: 'rising', filler: null }
        ],
        after: [],
        sentence_position: 'middle',
        laugh_within_s: 10.5
      },
      creator: {
        action: 'kept'
      },
      join: {
        gap_left_s: 0.0,
        flags: ['splice']
      }
    }
  ]
};

test('valid send passes validation', () => {
  const err = validateSend(validSend);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('unknown key at root rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.unknown_field = 'value';
  const err = validateSend(send);
  assert(err && err.includes('Unknown key'), `Expected unknown key error, got: ${err}`);
});

test('bad enum value rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].pace = 'invalid_pace';
  const err = validateSend(send);
  assert(err && err.includes('pace invalid'), `Expected pace enum error, got: ${err}`);
});

test('transcript word smuggled into pace rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].pace = 'custom_transcript';
  const err = validateSend(send);
  assert(err && err.includes('pace invalid'), `Expected pace invalid error, got: ${err}`);
});

test('transcript word smuggled into proposed.source rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].proposed.source = 'transcript';
  const err = validateSend(send);
  assert(err && err.includes('source invalid'), `Expected source invalid error, got: ${err}`);
});

test('too many records rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records = [];
  for (let i = 0; i < 501; i++) {
    send.records.push(validSend.records[0]);
  }
  const err = validateSend(send);
  assert(err && err.includes('exceeds 500'), `Expected exceeds 500 error, got: ${err}`);
});

test('bad send_id format rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.send_id = 'tooshort';
  const err = validateSend(send);
  assert(err && err.includes('send_id'), `Expected send_id error, got: ${err}`);
});

test('send_id with invalid chars rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.send_id = 'aBcDeFgHiJkLmNoPqRsT!';
  const err = validateSend(send);
  assert(err && err.includes('send_id'), `Expected send_id error, got: ${err}`);
});

test('bad plugin_version format rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.plugin_version = '0.2';
  const err = validateSend(send);
  assert(err && err.includes('plugin_version'), `Expected plugin_version error, got: ${err}`);
});

test('length_s out of range rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].proposed.length_s = 3601;
  const err = validateSend(send);
  assert(err && err.includes('length_s'), `Expected length_s error, got: ${err}`);
});

test('words out of range rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].proposed.words = 10001;
  const err = validateSend(send);
  assert(err && err.includes('words'), `Expected words error, got: ${err}`);
});

test('words not integer rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].proposed.words = 5.5;
  const err = validateSend(send);
  assert(err && err.includes('words'), `Expected words integer error, got: ${err}`);
});

test('unrounded length_s rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].proposed.length_s = 1.234;
  const err = validateSend(send);
  assert(err && err.includes('length_s'), `Expected length_s rounding error, got: ${err}`);
});

test('before array exceeds 3 rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before.push({ pos: 'noun', dur: 0.5, gap_after: 0.1, pitch: 'rising', filler: null });
  send.records[0].context.before.push({ pos: 'noun', dur: 0.5, gap_after: 0.1, pitch: 'rising', filler: null });
  send.records[0].context.before.push({ pos: 'noun', dur: 0.5, gap_after: 0.1, pitch: 'rising', filler: null });
  send.records[0].context.before.push({ pos: 'noun', dur: 0.5, gap_after: 0.1, pitch: 'rising', filler: null });
  const err = validateSend(send);
  assert(err && err.includes('before exceeds 3'), `Expected before exceeds 3 error, got: ${err}`);
});

test('unknown creator action rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].creator.action = 'invalid_action';
  const err = validateSend(send);
  assert(err && err.includes('creator.action'), `Expected creator.action error, got: ${err}`);
});

test('unknown flag rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].join.flags = ['invalid_flag'];
  const err = validateSend(send);
  assert(err && err.includes('flags'), `Expected flags error, got: ${err}`);
});

test('unknown key in word shape rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].unknown_key = 'value';
  const err = validateSend(send);
  assert(err && err.includes('unknown key'), `Expected unknown key error, got: ${err}`);
});

test('unknown key in proposed rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].proposed.unknown_field = 'value';
  const err = validateSend(send);
  assert(err && err.includes('unknown key'), `Expected unknown key error, got: ${err}`);
});

test('gap_left_s out of range rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].join.gap_left_s = 61;
  const err = validateSend(send);
  assert(err && err.includes('gap_left_s'), `Expected gap_left_s error, got: ${err}`);
});

test('dur out of range rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].dur = 11;
  const err = validateSend(send);
  assert(err && err.includes('dur'), `Expected dur error, got: ${err}`);
});

test('gap_after out of range rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].gap_after = 61;
  const err = validateSend(send);
  assert(err && err.includes('gap_after'), `Expected gap_after error, got: ${err}`);
});

test('laugh_within_s out of range rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.laugh_within_s = 601;
  const err = validateSend(send);
  assert(err && err.includes('laugh_within_s'), `Expected laugh_within_s error, got: ${err}`);
});

test('valid send with null laugh_within_s passes', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.laugh_within_s = null;
  const err = validateSend(send);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('valid send with null gap_left_s passes', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].join.gap_left_s = null;
  const err = validateSend(send);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('valid send with null filler passes', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].filler = null;
  const err = validateSend(send);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('valid filler enum value passes', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].filler = 'um';
  const err = validateSend(send);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('invalid filler value rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].filler = 'invalid_filler';
  const err = validateSend(send);
  assert(err && err.includes('filler'), `Expected filler error, got: ${err}`);
});

test('bad POS enum rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].pos = 'invalid_pos';
  const err = validateSend(send);
  assert(err && err.includes('pos'), `Expected pos error, got: ${err}`);
});

test('bad PITCH enum rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.before[0].pitch = 'invalid_pitch';
  const err = validateSend(send);
  assert(err && err.includes('pitch'), `Expected pitch error, got: ${err}`);
});

test('bad sentence_position enum rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records[0].context.sentence_position = 'invalid_position';
  const err = validateSend(send);
  assert(err && err.includes('sentence_position'), `Expected sentence_position error, got: ${err}`);
});

test('send_id at upper length limit passes', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.send_id = 'A'.repeat(32);
  const err = validateSend(send);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('send_id below minimum length rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.send_id = 'A'.repeat(15);
  const err = validateSend(send);
  assert(err && err.includes('send_id'), `Expected send_id error, got: ${err}`);
});

test('schema not 1 rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.schema = 2;
  const err = validateSend(send);
  assert(err && err.includes('schema'), `Expected schema error, got: ${err}`);
});

test('records not array rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records = 'not an array';
  const err = validateSend(send);
  assert(err && err.includes('records'), `Expected records error, got: ${err}`);
});

test('multiple records with one bad record rejected', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records.push(JSON.parse(JSON.stringify(validSend.records[0])));
  send.records[1].pace = 'invalid';
  const err = validateSend(send);
  assert(err && err.includes('Record 1'), `Expected Record 1 error, got: ${err}`);
});

test('empty records array passes', () => {
  const send = JSON.parse(JSON.stringify(validSend));
  send.records = [];
  const err = validateSend(send);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

// Feedback validation tests
const validFeedback = {
  schema: 1,
  feedback_id: 'aBcDeFgHiJkLmNoPqRsT',
  plugin_version: '0.2.0',
  message: 'This is great feedback',
  name: 'John Doe',
  email: 'john@example.com'
};

test('valid feedback with name and email passes', () => {
  const err = validateFeedback(validFeedback);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('valid feedback without name and email passes', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.name = null;
  fb.email = null;
  const err = validateFeedback(fb);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('valid feedback with only name passes', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.email = null;
  const err = validateFeedback(fb);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('valid feedback with only email passes', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.name = null;
  const err = validateFeedback(fb);
  assert.strictEqual(err, null, `Expected null, got: ${err}`);
});

test('bad email format rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.email = 'not-an-email';
  const err = validateFeedback(fb);
  assert(err && err.includes('email'), `Expected email error, got: ${err}`);
});

test('email without @ rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.email = 'johnexample.com';
  const err = validateFeedback(fb);
  assert(err && err.includes('email'), `Expected email error, got: ${err}`);
});

test('message too long rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.message = 'x'.repeat(5001);
  const err = validateFeedback(fb);
  assert(err && err.includes('message'), `Expected message error, got: ${err}`);
});

test('message too short rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.message = '';
  const err = validateFeedback(fb);
  assert(err && err.includes('message'), `Expected message error, got: ${err}`);
});

test('name exceeds 100 chars rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.name = 'x'.repeat(101);
  const err = validateFeedback(fb);
  assert(err && err.includes('name'), `Expected name error, got: ${err}`);
});

test('email exceeds 200 chars rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.email = 'a@b.com' + 'x'.repeat(200);
  const err = validateFeedback(fb);
  assert(err && err.includes('email'), `Expected email error, got: ${err}`);
});

test('unknown key in feedback rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.unknown_field = 'value';
  const err = validateFeedback(fb);
  assert(err && err.includes('Unknown key'), `Expected unknown key error, got: ${err}`);
});

test('bad feedback_id format rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.feedback_id = 'tooshort';
  const err = validateFeedback(fb);
  assert(err && err.includes('feedback_id'), `Expected feedback_id error, got: ${err}`);
});

test('bad plugin_version in feedback rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.plugin_version = '0.2';
  const err = validateFeedback(fb);
  assert(err && err.includes('plugin_version'), `Expected plugin_version error, got: ${err}`);
});

test('schema not 1 in feedback rejected', () => {
  const fb = JSON.parse(JSON.stringify(validFeedback));
  fb.schema = 2;
  const err = validateFeedback(fb);
  assert(err && err.includes('schema'), `Expected schema error, got: ${err}`);
});
