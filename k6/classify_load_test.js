import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { Trend, Rate } from 'k6/metrics';

// Load comments generated from classify_csv.ipynb data source
const comments = new SharedArray('comments', function () {
  return JSON.parse(open('./comments.json'));
});

const BASE_URL = __ENV.BASE_URL || 'http://127.0.0.1:8000';
const MAX_POLL_ATTEMPTS = 135;
const POLL_INTERVAL_SECONDS = 1;

// Custom metrics
// classify_submission_ms: time to POST /classify and receive a request_id (enqueue)
const classifySubmission = new Trend('classify_submission_ms');
const pollAttempts = new Trend('poll_attempts');
// total_latency_ms: true end-to-end time from the start of the classify request
// until the final result is returned (submission + polling)
const totalLatency = new Trend('total_latency_ms');
const resultErrorRate = new Rate('result_errors');

export const options = {
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<10000'],
    // Only the submission time; expect it to be fast because the endpoint is async
    classify_submission_ms: ['p(95)<5000'],
    // This is the metric that actually matters for user experience
    total_latency_ms: ['p(95)<150000'],
    result_errors: ['rate<0.05'],
  },
  scenarios: {
    // Moderate load: 5 VUs sustains mild queue pressure on a batch-4 CPU worker.
    // With ~100s inference latency, a 135s timeout lets most through but
    // ~20% of iterations hit the ceiling due to batching delays / queue depth.
    classify_ramp: {
      executor: 'ramping-vus',
      startVUs: 2,
      stages: [
        { duration: '20s', target: 4 },
        { duration: '40s', target: 6 },
        { duration: '216s', target: 6 },   // sustained moderate load
        { duration: '40s', target: 4 },
        { duration: '20s', target: 0 },
      ],
      gracefulRampDown: '120s',
    },
  },
};

function pickComment() {
  const idx = Math.floor(Math.random() * comments.length);
  return comments[idx];
}

export default function () {
  const comment = pickComment();

  // 1. Submit comment for classification
  const totalStart = Date.now();
  const classifyStart = Date.now();
  const classifyRes = http.post(
    `${BASE_URL}/classify`,
    JSON.stringify({ comment }),
    {
      headers: { 'Content-Type': 'application/json' },
      tags: { name: 'classify' },
    }
  );
  classifySubmission.add(Date.now() - classifyStart);

  const classifyOk = check(classifyRes, {
    'classify status is 200': (r) => r.status === 200,
    'classify returns request_id': (r) => {
      const body = r.json();
      return body && typeof body.request_id === 'string';
    },
  });

  if (!classifyOk || classifyRes.status !== 200) {
    resultErrorRate.add(1);
    return;
  }

  const requestId = classifyRes.json().request_id;

  // 2. Synchronously poll for the result
  let resultRes;
  let attempts = 0;
  let pending = true;
  const pollStart = Date.now();

  while (true) {
    resultRes = http.get(`${BASE_URL}/result/${requestId}`, {
      tags: { name: 'result' },
    });
    attempts++;

    if (resultRes.status === 200) {
      const body = resultRes.json();
      pending = body && body.status === 'pending';
    }

    if (!pending || attempts >= MAX_POLL_ATTEMPTS) {
      break;
    }

    sleep(POLL_INTERVAL_SECONDS);
  }

  const totalPollTime = Date.now() - pollStart;
  const totalTime = Date.now() - totalStart;
  totalLatency.add(totalTime);
  pollAttempts.add(attempts);

  const resultOk = check(resultRes, {
    'result status is 200': (r) => r.status === 200,
    'result is not pending': () => !pending,
    'result has non-empty label': (r) => {
      const body = r.json();
      return body && typeof body.label === 'string' && body.label !== '';
    },
    'result has valid score': (r) => {
      const body = r.json();
      return body && typeof body.score === 'number' && body.score > 0;
    },
    'result is not error': (r) => {
      const body = r.json();
      return body && body.status !== 'error';
    },
  });

  if (resultOk) {
    resultErrorRate.add(0);
    console.log(
      `OK | request_id=${requestId} | total_latency=${totalTime}ms | poll_time=${totalPollTime}ms | attempts=${attempts} | comment="${comment.substring(0, 60)}..." | label=${resultRes.json().label} | score=${resultRes.json().score}`
    );
  } else if (pending) {
    resultErrorRate.add(1);
    console.log(
      `TIMEOUT | request_id=${requestId} | total_latency=${totalTime}ms | poll_time=${totalPollTime}ms | attempts=${attempts} | comment="${comment.substring(0, 60)}..."`
    );
  } else {
    resultErrorRate.add(1);
    let body;
    try {
      body = resultRes.json();
    } catch (e) {
      body = {};
    }
    console.log(
      `ERROR | request_id=${requestId} | status=${body.status || resultRes.status || 'unknown'} | comment="${comment.substring(0, 60)}..."`
    );
  }
}
