/*
  Quantum Teleportation Simulator  (Single-Qubit Teleportation, Three-Qubit Protocol)
  Platform : Wokwi  (Arduino Uno + 2.8" ILI9341 TFT + pot + 5 buttons + 2 LEDs)
  Author   : Sarvesh Salvi

  HOW IT WORKS
  - A real 3-qubit STATE-VECTOR simulation (8 complex amplitudes) runs on the Uno.
      qubit 0 = |psi> (Alice's unknown state), qubit 1 = Alice's half of the Bell pair,
      qubit 2 = Bob's half of the Bell pair.
  - Circuit: H + CNOT make the Bell pair, then Alice does CNOT(0,1) + H(0),
    measures qubits 0 and 1 -> 2 classical bits -> Bob applies X / Z correction.
  - NOISE (Monte-Carlo trajectories):
      1. Depolarizing : with probability = noise, Bob's half gets a random Pauli (I,X,Y,Z)
      2. Qubit loss   : the qubit is lost with probability 10 %
      3. Bit-flip     : each classical bit flips with probability 5 %
  - NOISE REDUCTION (button A2):
      a) Entanglement purification (2 rounds, BBPSSW formula)  -> better Bell pair
      b) Classical bits sent 3 times + majority vote           -> fixes bit-flips
      c) Alice re-sends a lost qubit up to 3 times             -> fixes loss
  - Fidelity F = |<psi|Bob>|^2.  Classical limit = 2/3.  Success = qubit arrived AND F > 2/3.

  CONTROLS
    Potentiometer A0 : noise level 0..100 %
    D2 : change input state (|0>, |1>, |+>, |->)
    D3 : START teleportation (100 sends)  / back to dashboard
    D4 : GRAPH  - sweep noise 0..100 % and plot fidelity + success rate
    D5 : change noise model (Depol / +Loss / +Flip / all)
    A2 : noise reduction ON / OFF
    LED green D6 : average fidelity above 2/3      LED red D7 : below 2/3
*/

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

// ---------------- pins ----------------
#define TFT_CS  10
#define TFT_DC  9
#define TFT_RST 8
const uint8_t PIN_POT    = A0;
const uint8_t PIN_STATE  = 2;
const uint8_t PIN_START  = 3;
const uint8_t PIN_GRAPH  = 4;
const uint8_t PIN_MODEL  = 5;
const uint8_t PIN_REDUCE = A2;
const uint8_t PIN_LED_OK = 6;
const uint8_t PIN_LED_BAD = 7;

Adafruit_ILI9341 tft(TFT_CS, TFT_DC, TFT_RST);

// flash-string print helpers (save the Uno's tiny RAM)
#define TP(s)  tft.print(F(s))
#define SP(s)  Serial.print(F(s))
#define SPL(s) Serial.println(F(s))

// ---------------- simulation settings ----------------
const float LOSS_P  = 0.10f;      // qubit loss probability (Loss model)
const float FLIP_P  = 0.05f;      // classical bit-flip probability (Flip model)
const float LIMIT   = 0.6667f;    // classical limit 2/3
const uint8_t N_SEND  = 100;      // sends per START
const uint8_t N_SWEEP = 150;      // sends per point in the graph sweep
const float S2 = 0.70710678f;

// ---------------- complex numbers + gates ----------------
struct Cx { float r, i; };
static inline Cx cx(float r, float i) { Cx c; c.r = r; c.i = i; return c; }
static inline Cx cmul(Cx a, Cx b) { return cx(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
static inline Cx cadd(Cx a, Cx b) { return cx(a.r + b.r, a.i + b.i); }
static inline float cabs2(Cx a) { return a.r * a.r + a.i * a.i; }

// 2x2 gates stored row by row: {m00, m01, m10, m11}
const Cx G_I[4] = { {1, 0}, {0, 0}, {0, 0}, {1, 0} };
const Cx G_X[4] = { {0, 0}, {1, 0}, {1, 0}, {0, 0} };
const Cx G_Y[4] = { {0, 0}, {0, -1}, {0, 1}, {0, 0} };
const Cx G_Z[4] = { {1, 0}, {0, 0}, {0, 0}, {-1, 0} };
const Cx G_H[4] = { {S2, 0}, {S2, 0}, {S2, 0}, {-S2, 0} };
const Cx *PAULI[4] = { G_I, G_X, G_Y, G_Z };

Cx st[8];   // 3-qubit state vector, index = 4*q0 + 2*q1 + q2

void apply1(uint8_t q, const Cx *g) {           // single-qubit gate on qubit q (0,1,2)
  uint8_t m = 1 << (2 - q);
  for (uint8_t k = 0; k < 8; k++) {
    if (k & m) continue;
    Cx a = st[k], b = st[k | m];
    st[k]     = cadd(cmul(g[0], a), cmul(g[1], b));
    st[k | m] = cadd(cmul(g[2], a), cmul(g[3], b));
  }
}

void cnot(uint8_t c, uint8_t t) {               // controlled-NOT
  uint8_t mc = 1 << (2 - c), mt = 1 << (2 - t);
  for (uint8_t k = 0; k < 8; k++) {
    if ((k & mc) && !(k & mt)) { Cx tmp = st[k]; st[k] = st[k | mt]; st[k | mt] = tmp; }
  }
}

float rand01() { return random(0, 10000) / 10000.0f; }

// ---------------- settings chosen by the user ----------------
uint8_t stateIdx = 2;        // 0:|0>  1:|1>  2:|+>  3:|->
uint8_t modelIdx = 3;        // 0:Depol  1:+Loss  2:+Flip  3:all
bool reduceOn = false;
float noise = 0.30f;         // 0..1 from the potentiometer
float lastF = 0, lastS = 0;
bool haveResult = false;
uint8_t screenMode = 0;      // 0 = dashboard, 1 = graph screen
bool prevBtn[5] = { false, false, false, false, false };

bool useLoss() { return modelIdx == 1 || modelIdx == 3; }
bool useFlip() { return modelIdx == 2 || modelIdx == 3; }

void psiAmps(uint8_t s, float &a, float &b) {
  if (s == 0)      { a = 1;  b = 0;  }
  else if (s == 1) { a = 0;  b = 1;  }
  else if (s == 2) { a = S2; b = S2; }
  else             { a = S2; b = -S2; }
}

// send one classical bit over the noisy classical channel (optional 3x repetition)
uint8_t sendBit(uint8_t bit, float flipP, uint8_t reps) {
  uint8_t ones = 0;
  for (uint8_t r = 0; r < reps; r++) {
    uint8_t b = bit;
    if (rand01() < flipP) b ^= 1;
    ones += b;
  }
  return (ones * 2 > reps) ? 1 : 0;
}

// noise reduction (a): entanglement purification, 2 rounds of the BBPSSW formula.
// Works only while the pair fidelity is above 1/2 (noise below about 67 %).
float effLambda(float lam, bool reduce) {
  if (!reduce) return lam;
  float Fp = 1.0f - 0.75f * lam;                // Bell-pair fidelity of the noisy pair
  for (uint8_t round = 0; round < 2; round++) {
    if (Fp <= 0.5f) break;
    float e = (1.0f - Fp) / 3.0f;
    float num = Fp * Fp + e * e;
    float den = Fp * Fp + 2.0f * Fp * e + 5.0f * e * e;
    Fp = num / den;
  }
  return (1.0f - Fp) * 4.0f / 3.0f;
}

// ONE teleportation attempt. Returns fidelity, or -1 if the qubit was lost.
float oneTrial(float lamEff, bool loss, bool flip, bool reduce) {
  if (loss) {                                    // qubit loss (+ re-send when reducing)
    uint8_t tries = reduce ? 3 : 1;
    bool arrived = false;
    for (uint8_t t = 0; t < tries && !arrived; t++) arrived = (rand01() >= LOSS_P);
    if (!arrived) return -1.0f;
  }

  float a, b;
  psiAmps(stateIdx, a, b);
  for (uint8_t k = 0; k < 8; k++) st[k] = cx(0, 0);
  st[0] = cx(a, 0);                              // |psi> (x) |00>
  st[4] = cx(b, 0);

  apply1(1, G_H);                                // make the Bell pair on qubits 1,2
  cnot(1, 2);
  if (rand01() < lamEff)                         // depolarizing noise on Bob's half
    apply1(2, PAULI[random(0, 4)]);

  cnot(0, 1);                                    // Alice's Bell-measurement circuit
  apply1(0, G_H);

  float p[4];                                    // measure qubits 0,1 -> outcome o
  for (uint8_t o = 0; o < 4; o++) p[o] = cabs2(st[2 * o]) + cabs2(st[2 * o + 1]);
  float r = rand01(), acc = 0;
  uint8_t o = 0;
  for (uint8_t k = 0; k < 4; k++) { if (p[k] > p[o]) o = k; }   // safe default
  for (uint8_t k = 0; k < 4; k++) { acc += p[k]; if (r < acc) { o = k; break; } }

  float inv = 1.0f / sqrtf(p[o]);                // Bob's qubit after the measurement
  Cx b0 = cx(st[2 * o].r * inv, st[2 * o].i * inv);
  Cx b1 = cx(st[2 * o + 1].r * inv, st[2 * o + 1].i * inv);

  uint8_t reps = reduce ? 3 : 1;                 // classical channel
  float fp = flip ? FLIP_P : 0.0f;
  uint8_t r0 = sendBit(o >> 1, fp, reps);
  uint8_t r1 = sendBit(o & 1, fp, reps);

  if (r1) { Cx t = b0; b0 = b1; b1 = t; }        // Bob: X if r1
  if (r0) b1 = cx(-b1.r, -b1.i);                 // Bob: Z if r0

  Cx ov = cadd(cmul(cx(a, 0), b0), cmul(cx(b, 0), b1));   // <psi|Bob>
  return cabs2(ov);
}

// Monte Carlo: n sends -> mean fidelity (of arrived qubits) and success rate
void runTrials(uint8_t n, float lam, bool loss, bool flip, bool reduce, float &meanF, float &succ) {
  float le = effLambda(lam, reduce);
  float sum = 0;
  uint8_t arrived = 0, ok = 0;
  for (uint8_t i = 0; i < n; i++) {
    float f = oneTrial(le, loss, flip, reduce);
    if (f >= 0) { sum += f; arrived++; if (f > LIMIT) ok++; }
  }
  meanF = arrived ? sum / arrived : 0.0f;
  succ = (float)ok / n;
}

// ---------------- drawing helpers ----------------
void panel(int x, int y, int w, int h, uint16_t col) {
  tft.drawRect(x, y, w, h, col);
  tft.fillRect(x, y, w, 12, col);
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_BLACK, col);
  tft.setCursor(x + 4, y + 2);
}

void printState(uint8_t s) {
  if (s == 0) TP("|0>");
  else if (s == 1) TP("|1>");
  else if (s == 2) TP("|+>");
  else TP("|->");
}

void printModel() {
  if (modelIdx == 0) TP("Depolarizing only    ");
  else if (modelIdx == 1) TP("Depol + Qubit loss   ");
  else if (modelIdx == 2) TP("Depol + Bit-flip     ");
  else TP("Depol + Loss + Flip ");
}

void updateAlice() {
  tft.setTextSize(3);
  tft.setTextColor(ILI9341_YELLOW, ILI9341_BLACK);
  tft.setCursor(14, 62);
  printState(stateIdx);
}

void updateBob() {
  tft.setTextSize(3);
  tft.setCursor(222, 62);
  if (haveResult) {
    tft.setTextColor(lastF > LIMIT ? ILI9341_GREEN : ILI9341_RED, ILI9341_BLACK);
    tft.print(lastF, 2);
  } else {
    tft.setTextColor(ILI9341_DARKGREY, ILI9341_BLACK);
    TP("----");
  }
  tft.setTextSize(2);
  tft.setCursor(222, 98);
  if (haveResult) {
    tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
    tft.print((int)(lastS * 100.0f + 0.5f));
    TP("%  ");
  } else {
    tft.setTextColor(ILI9341_DARKGREY, ILI9341_BLACK);
    TP("--  ");
  }
  tft.setTextSize(1);
  tft.setCursor(222, 120);
  if (!haveResult) { tft.setTextColor(ILI9341_DARKGREY, ILI9341_BLACK); TP("press START   "); }
  else if (lastF > LIMIT) { tft.setTextColor(ILI9341_GREEN, ILI9341_BLACK); TP("SUCCESS!      "); }
  else { tft.setTextColor(ILI9341_RED, ILI9341_BLACK); TP("BELOW 2/3 LIMIT"); }
}

void updateNoise() {
  tft.drawRect(80, 152, 170, 10, ILI9341_WHITE);
  int w = (int)(noise * 166.0f);
  tft.fillRect(82, 154, w, 6, ILI9341_ORANGE);
  tft.fillRect(82 + w, 154, 166 - w, 6, ILI9341_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(ILI9341_ORANGE, ILI9341_BLACK);
  tft.setCursor(260, 150);
  tft.print((int)(noise * 100.0f + 0.5f));
  TP("%  ");
}

void updateModelReduce() {
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(8, 170);
  TP("Model [D5]: ");
  printModel();
  tft.setCursor(8, 181);
  TP("Reduce [A2]: ");
  if (reduceOn) { tft.setTextColor(ILI9341_GREEN, ILI9341_BLACK); TP("ON  (purify x2, 3x bits, resend x3)"); }
  else { tft.setTextColor(ILI9341_LIGHTGREY, ILI9341_BLACK); TP("OFF                                 "); }
}

void updateInfo() {
  tft.setTextSize(1);
  tft.setCursor(8, 218);
  if (!haveResult) { tft.setTextColor(ILI9341_LIGHTGREY, ILI9341_BLACK); TP("Press START to teleport 100 qubits.                "); }
  else if (lastF > LIMIT) { tft.setTextColor(ILI9341_GREEN, ILI9341_BLACK); TP("F > 2/3 : entangled link beats classical.         "); }
  else { tft.setTextColor(ILI9341_RED, ILI9341_BLACK); TP("F < 2/3 : classical wins - cut noise / reduce!"); }
  digitalWrite(PIN_LED_OK, haveResult && lastF > LIMIT);
  digitalWrite(PIN_LED_BAD, haveResult && lastF <= LIMIT);
}

void drawDashboard() {
  screenMode = 0;
  tft.fillScreen(ILI9341_BLACK);
  tft.fillRect(0, 0, 320, 20, ILI9341_NAVY);
  tft.setTextSize(2);
  tft.setTextColor(ILI9341_WHITE, ILI9341_NAVY);
  tft.setCursor(14, 3);
  TP("QUANTUM TELEPORTATION");
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_CYAN, ILI9341_BLACK);
  tft.setCursor(43, 24);
  TP("Single qubit - 3-qubit protocol (Wokwi)");

  // 1. Alice
  panel(2, 36, 104, 96, ILI9341_CYAN); TP("1 ALICE (Tx)");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(8, 52); TP("Input state:");
  tft.setCursor(8, 92); TP("[D2] change state");
  tft.setCursor(8, 103); TP("[D3] START");
  tft.setCursor(8, 114); TP("[D4] graph sweep");

  // 2. Quantum circuit
  panel(108, 36, 104, 96, ILI9341_MAGENTA); TP("2 QUANTUM CIRCUIT");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.drawFastHLine(114, 62, 92, ILI9341_WHITE);          // qubit 0 wire (|psi>)
  tft.drawFastHLine(114, 86, 92, ILI9341_WHITE);          // qubit 1 wire (Alice's half)
  tft.setCursor(112, 53); TP("psi");
  tft.setCursor(112, 77); TP("A");
  tft.drawFastVLine(142, 62, 24, ILI9341_WHITE);          // CNOT link
  tft.fillCircle(142, 62, 3, ILI9341_WHITE);              // control
  tft.drawCircle(142, 86, 5, ILI9341_WHITE);              // target (+)
  tft.drawFastHLine(138, 86, 9, ILI9341_WHITE);
  tft.drawRect(156, 54, 16, 16, ILI9341_YELLOW);          // H gate
  tft.setCursor(161, 58); tft.setTextColor(ILI9341_YELLOW, ILI9341_BLACK); TP("H");
  tft.drawRect(184, 54, 16, 16, ILI9341_CYAN);            // measure
  tft.drawRect(184, 78, 16, 16, ILI9341_CYAN);
  tft.setTextColor(ILI9341_CYAN, ILI9341_BLACK);
  tft.setCursor(189, 58); TP("M");
  tft.setCursor(189, 82); TP("M");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(112, 103); TP("Bell pair + 2 bits");
  tft.setCursor(112, 114); TP("Bob fixes: X / Z");

  // 3. Bob
  panel(214, 36, 104, 96, ILI9341_GREEN); TP("3 BOB (Rx)");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(220, 52); TP("Avg fidelity:");
  tft.setCursor(220, 88); TP("Success rate:");

  // 6. Noise control
  panel(2, 136, 316, 56, ILI9341_ORANGE); TP("6 NOISE CONTROL  (turn the potentiometer)");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(8, 153); TP("Noise level");

  // 7. Info
  panel(2, 196, 316, 42, ILI9341_LIGHTGREY); TP("7 RESULT");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(8, 229); TP("Limit: classical = 2/3.  Green LED OK, red LED low.");

  updateAlice(); updateBob(); updateNoise(); updateModelReduce(); updateInfo();
}

// ---------------- sweep + graph screen ----------------
float fB[11], sB[11], fR[11], sR[11];

const int GX1 = 34, GX2 = 194, GY = 32, GW = 112, GH = 96;

int pxOf(int x0, int i) { return x0 + (i * GW) / 10; }
int pyF(float v) { if (v < 0.4f) v = 0.4f; if (v > 1.0f) v = 1.0f; return GY + GH - (int)((v - 0.4f) / 0.6f * GH); }
int pyS(float v) { if (v < 0) v = 0; if (v > 1) v = 1; return GY + GH - (int)(v * GH); }

void curve(int x0, float *v, bool isF, uint16_t col) {
  for (uint8_t i = 0; i < 11; i++) {
    int y = isF ? pyF(v[i]) : pyS(v[i]);
    if (i > 0) {
      int yp = isF ? pyF(v[i - 1]) : pyS(v[i - 1]);
      tft.drawLine(pxOf(x0, i - 1), yp, pxOf(x0, i), y, col);
    }
    tft.fillCircle(pxOf(x0, i), y, 2, col);
  }
}

float crossing(float *f) {                       // noise % where F falls below 2/3
  if (f[0] < LIMIT) return 0;
  for (uint8_t i = 1; i < 11; i++) {
    if (f[i] < LIMIT) {
      float t = (f[i - 1] - LIMIT) / (f[i - 1] - f[i]);
      return ((i - 1) + t) * 10.0f;
    }
  }
  return -1;
}

void runSweep() {
  screenMode = 1;
  tft.fillScreen(ILI9341_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(ILI9341_YELLOW, ILI9341_BLACK);
  tft.setCursor(20, 90); TP("Running noise sweep...");
  tft.drawRect(40, 130, 240, 14, ILI9341_WHITE);
  for (uint8_t i = 0; i < 11; i++) {
    float n = i / 10.0f;
    runTrials(N_SWEEP, n, useLoss(), useFlip(), false, fB[i], sB[i]);
    runTrials(N_SWEEP, n, useLoss(), useFlip(), true, fR[i], sR[i]);
    tft.fillRect(42, 132, (i + 1) * 236 / 11, 10, ILI9341_GREEN);
  }
  // serial table for the report
  SPL("noise%,F_baseline,success%_baseline,F_reduced,success%_reduced");
  for (uint8_t i = 0; i < 11; i++) {
    Serial.print(i * 10); Serial.print(',');
    Serial.print(fB[i], 3); Serial.print(',');
    Serial.print(sB[i] * 100.0f, 0); Serial.print(',');
    Serial.print(fR[i], 3); Serial.print(',');
    Serial.println(sR[i] * 100.0f, 0);
  }
}

void drawGraphScreen() {
  tft.fillScreen(ILI9341_BLACK);
  tft.fillRect(0, 0, 320, 14, ILI9341_NAVY);
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_WHITE, ILI9341_NAVY);
  tft.setCursor(8, 3);
  TP("FIDELITY & SUCCESS RATE vs NOISE  (state ");
  printState(stateIdx);
  TP(")");

  tft.setTextColor(ILI9341_CYAN, ILI9341_BLACK);
  tft.setCursor(GX1, 19); TP("4 Fidelity vs noise");
  tft.setCursor(GX2, 19); TP("5 Success rate vs noise");

  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.drawRect(GX1, GY, GW, GH, ILI9341_WHITE);
  tft.drawRect(GX2, GY, GW, GH, ILI9341_WHITE);
  tft.drawFastVLine(GX1 + GW / 2, GY, GH, ILI9341_DARKGREY);
  tft.drawFastVLine(GX2 + GW / 2, GY, GH, ILI9341_DARKGREY);
  tft.drawFastHLine(GX1, GY + GH / 2, GW, ILI9341_DARKGREY);
  tft.drawFastHLine(GX2, GY + GH / 2, GW, ILI9341_DARKGREY);
  tft.setCursor(GX1 - 20, GY - 3);          TP("1.0");
  tft.setCursor(GX1 - 20, GY + GH - 5);     TP("0.4");
  tft.setCursor(GX2 - 26, GY - 3);          TP("100%");
  tft.setCursor(GX2 - 14, GY + GH - 5);     TP("0%");
  tft.setCursor(GX1 - 2, GY + GH + 3);      TP("0");
  tft.setCursor(GX1 + GW / 2 - 6, GY + GH + 3); TP("50");
  tft.setCursor(GX1 + GW - 10, GY + GH + 3);    TP("100");
  tft.setCursor(GX2 - 2, GY + GH + 3);      TP("0");
  tft.setCursor(GX2 + GW / 2 - 6, GY + GH + 3); TP("50");
  tft.setCursor(GX2 + GW - 10, GY + GH + 3);    TP("100");
  tft.setCursor(GX1 + 28, GY + GH + 13);    TP("Noise p (%)");
  tft.setCursor(GX2 + 28, GY + GH + 13);    TP("Noise p (%)");

  int yl = pyF(LIMIT);                      // dashed classical-limit line
  for (int x = GX1; x < GX1 + GW; x += 6) tft.drawFastHLine(x, yl, 3, ILI9341_RED);
  tft.setTextColor(ILI9341_RED, ILI9341_BLACK);
  tft.setCursor(GX1 + GW + 3, yl - 3); TP("2/3");

  curve(GX1, fB, true, ILI9341_LIGHTGREY);
  curve(GX1, fR, true, ILI9341_GREEN);
  curve(GX2, sB, false, ILI9341_LIGHTGREY);
  curve(GX2, sR, false, ILI9341_GREEN);

  tft.drawFastHLine(8, 158, 18, ILI9341_LIGHTGREY);
  tft.setTextColor(ILI9341_LIGHTGREY, ILI9341_BLACK); tft.setCursor(30, 155); TP("Baseline");
  tft.drawFastHLine(100, 158, 18, ILI9341_GREEN);
  tft.setTextColor(ILI9341_GREEN, ILI9341_BLACK); tft.setCursor(122, 155); TP("Noise reduction ON");
  for (int x = 238; x < 258; x += 6) tft.drawFastHLine(x, 158, 3, ILI9341_RED);
  tft.setTextColor(ILI9341_RED, ILI9341_BLACK); tft.setCursor(262, 155); TP("2/3 limit");

  // interpretation (computed from the sweep)
  tft.setTextColor(ILI9341_YELLOW, ILI9341_BLACK);
  tft.setCursor(8, 168);
  TP("Noise lowers F roughly as F = 1 - noise/2.");
  tft.setTextColor(ILI9341_WHITE, ILI9341_BLACK);
  tft.setCursor(8, 180);
  TP("30% noise: F "); tft.print(fB[3], 2); TP(" -> "); tft.print(fR[3], 2);
  TP("   success "); tft.print((int)(sB[3] * 100)); TP("% -> "); tft.print((int)(sR[3] * 100)); TP("%");
  tft.setCursor(8, 191);
  TP("60% noise: F "); tft.print(fB[6], 2); TP(" -> "); tft.print(fR[6], 2);
  TP("   success "); tft.print((int)(sB[6] * 100)); TP("% -> "); tft.print((int)(sR[6] * 100)); TP("%");
  tft.setCursor(8, 203);
  float c = crossing(fB);
  if (c < 0) { TP("Baseline stays above 2/3 for all noise."); }
  else { TP("Baseline falls below 2/3 at noise ~"); tft.print((int)(c + 0.5f)); TP("%"); }
  tft.setTextColor(ILI9341_CYAN, ILI9341_BLACK);
  tft.setCursor(8, 214);
  TP("Reduction needs pair F>1/2 (noise < ~67%):");
  tft.setCursor(8, 224);
  TP("beyond that purification cannot help. [D3] back");
}

// ---------------- actions ----------------
void doStart() {
  drawDashboard();
  tft.setTextSize(1);
  tft.setTextColor(ILI9341_YELLOW, ILI9341_BLACK);
  tft.setCursor(220, 120); TP("running...    ");
  runTrials(N_SEND, noise, useLoss(), useFlip(), reduceOn, lastF, lastS);
  haveResult = true;
  updateBob();
  updateInfo();
  SP("noise="); Serial.print((int)(noise * 100.0f + 0.5f));
  SP("%  reduce="); Serial.print(reduceOn ? F("ON") : F("OFF"));
  SP("  ->  F="); Serial.print(lastF, 3);
  SP("  success="); Serial.print((int)(lastS * 100.0f + 0.5f));
  SPL("%");
}

void doGraph() {
  runSweep();
  drawGraphScreen();
}

bool edge(uint8_t idx, uint8_t pin) {
  bool now = (digitalRead(pin) == LOW);
  bool e = now && !prevBtn[idx];
  prevBtn[idx] = now;
  return e;
}

void setup() {
  Serial.begin(9600);
  pinMode(PIN_STATE, INPUT_PULLUP);
  pinMode(PIN_START, INPUT_PULLUP);
  pinMode(PIN_GRAPH, INPUT_PULLUP);
  pinMode(PIN_MODEL, INPUT_PULLUP);
  pinMode(PIN_REDUCE, INPUT_PULLUP);
  pinMode(PIN_LED_OK, OUTPUT);
  pinMode(PIN_LED_BAD, OUTPUT);
  randomSeed(analogRead(A1));
  tft.begin();
  tft.setRotation(1);                            // 320 x 240 landscape
  noise = analogRead(PIN_POT) / 1023.0f;
  drawDashboard();
  SPL("Quantum teleportation simulator ready.");
}

void loop() {
  // live noise knob
  float n = (int)(analogRead(PIN_POT) / 1023.0f * 100.0f + 0.5f) / 100.0f;
  if (n != noise) { noise = n; if (screenMode == 0) updateNoise(); }

  if (edge(0, PIN_STATE)) {
    stateIdx = (stateIdx + 1) % 4; haveResult = false;
    drawDashboard();
  }
  if (edge(1, PIN_START)) doStart();
  if (edge(2, PIN_GRAPH)) doGraph();
  if (edge(3, PIN_MODEL)) {
    modelIdx = (modelIdx + 1) % 4; haveResult = false;
    drawDashboard();
  }
  if (edge(4, PIN_REDUCE)) {
    reduceOn = !reduceOn; haveResult = false;
    drawDashboard();
  }
  delay(20);
}
