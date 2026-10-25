import { ACRONYM, FOOTNOTE, KICKER, WORDMARK } from "../brand";
import Logo from "./Logo";

const STARTERS = [
  { index: "01", title: "Review my code", detail: "Find bugs and improvements", text: "Review this code and point out bugs and improvements:\n\n" },
  { index: "02", title: "Work through a problem", detail: "Break it down, then solve it", text: "Help me work through this problem step by step:\n\n" },
  { index: "03", title: "Explain an error", detail: "Understand what went wrong", text: "Explain this error and how to fix it:\n\n" },
  { index: "04", title: "Draft a message", detail: "Find the right words", text: "Write a polite, professional message to " },
];

function IntelligenceCore() {
  const points = [
    [220, 220, 5], [185, 174, 3], [254, 179, 3], [273, 221, 3.5], [244, 264, 3], [192, 269, 3],
    [151, 202, 2.5], [164, 143, 2], [210, 125, 2.5], [266, 133, 2], [306, 166, 2.5], [321, 217, 2],
    [300, 270, 2.5], [259, 306, 2], [204, 315, 2.5], [156, 283, 2], [125, 241, 2.5], [126, 181, 2],
    [188, 101, 2], [241, 91, 2], [291, 111, 2], [338, 149, 2], [349, 205, 2], [337, 260, 2],
    [302, 313, 2], [251, 343, 2], [195, 350, 2], [143, 327, 2], [104, 287, 2], [91, 231, 2],
    [99, 172, 2], [132, 124, 2], [179, 75, 2], [230, 62, 2], [282, 76, 2], [327, 99, 2],
    [364, 137, 2], [378, 190, 2], [371, 245, 2], [346, 298, 2], [309, 340, 2], [255, 371, 2],
    [197, 379, 2], [139, 361, 2], [91, 325, 2], [66, 273, 2], [61, 215, 2], [74, 159, 2],
  ];
  const links = [[0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,6],[1,7],[1,8],[2,3],[2,9],[2,10],[3,4],[3,11],[3,12],[4,5],[4,13],[4,14],[5,6],[5,15],[6,7],[6,16],[7,8],[7,17],[8,9],[8,18],[9,10],[9,19],[10,11],[10,20],[11,12],[11,21],[12,13],[12,22],[13,14],[13,23],[14,15],[14,24],[15,16],[15,25],[16,17],[16,26],[17,18],[18,19],[18,27],[19,20],[19,28],[20,21],[20,29],[21,22],[21,30],[22,23],[22,31],[23,24],[23,32],[24,25],[24,33],[25,26],[25,34],[26,27],[26,35],[27,28],[27,36],[28,29],[28,37],[29,30],[29,38],[30,31],[30,39],[31,32],[31,40],[32,33],[32,41],[33,34],[33,42],[34,35],[34,43],[35,36],[35,44],[36,37],[36,45],[37,38],[37,46],[38,39],[38,47],[39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47]];
  return (
    <div className="core-stage neuron-stage" aria-label="Atlas neural intelligence visualization">
      <div className="core-coordinate coordinate-top">ATLAS / NEURAL NETWORK <span>01—A</span></div>
      <svg className="core-visual neuron-visual" viewBox="0 0 440 440" role="img" aria-label="A network of connected, moving points">
        <defs><radialGradient id="neuronGlow"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".18"/><stop offset="100%" stopColor="var(--accent)" stopOpacity="0"/></radialGradient></defs>
        <circle cx="220" cy="220" r="190" fill="url(#neuronGlow)" />
        <g className="neuron-links">{links.map(([a,b], i) => <line key={i} x1={points[a][0]} y1={points[a][1]} x2={points[b][0]} y2={points[b][1]} style={{ animationDelay: `${(i % 13) * -0.37}s` }} />)}</g>
        <g className="neuron-points">{points.map(([x,y,r], i) => <circle key={i} cx={x} cy={y} r={r} className={i % 7 === 0 ? "neuron-dot bright" : "neuron-dot"} style={{ animationDelay: `${(i % 11) * -0.42}s` }} />)}</g>
        <circle className="neuron-heart" cx="220" cy="220" r="18" />
        <circle className="neuron-heart-core" cx="220" cy="220" r="4" />
      </svg>
      <div className="core-coordinate coordinate-left"><span className="core-live-dot" /> LOCAL INFERENCE <b>READY</b></div>
      <div className="core-coordinate coordinate-bottom">PRIVATE BY DESIGN <span>127.0.0.1</span></div>
      <div className="core-side-note">FIG. 01 <span>ADAPTIVE INTELLIGENCE</span></div>
    </div>
  );
}

export default function Welcome({ onPick }) {
  return (
    <section className="welcome">
      <div className="welcome-inner">
        <div className="welcome-eyebrow"><span className="eyebrow-line" /> PERSONAL INTELLIGENCE, ON YOUR TERMS <span className="eyebrow-number">A—001</span></div>
        <div className="welcome-hero">
          <div className="welcome-copy">
            <div className="welcome-brand"><Logo size={43} /><span>{WORDMARK}</span><i /></div>
            <p className="kicker">{KICKER}</p>
            <h1 className="wordmark-xl">Intelligence,<br /><em>in your hands.</em></h1>
            <p className="welcome-description">Your personal intelligence. Thoughtful, capable, and running on your terms.</p>
            <div className="welcome-identity"><span className="identity-rule" />{ACRONYM}</div>
          </div>
          <IntelligenceCore />
        </div>

        <div className="starters-panel compact-starters" aria-label="Suggested starting points">
          <div className="section-heading"><span>WHAT CAN WE WORK ON?</span><span>QUICK START</span></div>
          <ul className="starters">
            {STARTERS.map((s) => (
              <li key={s.title}>
                <button type="button" onClick={() => onPick(s.text)}>
                  <span className="starter-index">{s.index}</span>
                  <span className="starter-copy"><strong>{s.title}</strong><small>{s.detail}</small></span>
                  <i aria-hidden="true">↗</i>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <footer className="welcome-footer compact-footer"><span><Logo size={17} /> {FOOTNOTE}</span><span>DESIGNED TO BE USEFUL.</span></footer>
      </div>
    </section>
  );
}
