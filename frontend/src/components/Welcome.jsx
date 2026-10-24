import { ACRONYM, FOOTNOTE, KICKER, MOTTO, PRINCIPLES, WORDMARK } from "../brand";
import Logo from "./Logo";

const STARTERS = [
  { title: "Review my code", text: "Review this code and point out bugs and improvements:\n\n" },
  { title: "Total up a list", text: "Add up this list and show the total:\n\n" },
  { title: "Explain an error", text: "Explain this error and how to fix it:\n\n" },
  { title: "Draft a message", text: "Write a polite, professional message to " },
];

// A line-drawn globe, tilted by the Earth's axial tilt, with one marker: this machine.
function Globe() {
  const R = 270;
  const parallels = [-215, -150, -80, 0, 80, 150, 215];
  const half = (y) => Math.sqrt(R * R - y * y);

  return (
    <svg className="globe" viewBox="-310 -310 620 620" aria-hidden="true">
      <g transform="rotate(-23.4)">
        <circle r={R} />
        {parallels.map((y) => (
          <line key={y} x1={-half(y)} x2={half(y)} y1={y} y2={y} />
        ))}
        {[70, 140, 210].map((rx) => (
          <ellipse key={rx} rx={rx} ry={R} />
        ))}
        <line x1="0" x2="0" y1={-R - 30} y2={R + 30} />
      </g>
      <g className="globe-marker">
        <circle cx="168" cy="-150" r="13" className="ring" />
        <circle cx="168" cy="-150" r="4.5" className="dot" />
        <path d="M182 -150 H226" />
        <text x="236" y="-154">THIS MACHINE</text>
        <text x="236" y="-132">127.0.0.1</text>
      </g>
    </svg>
  );
}

export default function Welcome({ onPick }) {
  return (
    <section className="welcome">
      <Globe />
      <div className="welcome-inner">
        <Logo size={58} />
        <h1 className="wordmark-xl">{WORDMARK}</h1>
        <p className="kicker">{KICKER}</p>
        <p className="acronym">{ACRONYM}</p>

        <p className="motto">
          {MOTTO.map((line) => (
            <span key={line}>{line}</span>
          ))}
        </p>

        <ol className="principles">
          {PRINCIPLES.map((p) => (
            <li key={p.numeral}>
              <span className="numeral">{p.numeral}</span>
              <strong>{p.title}</strong>
              <span>{p.text}</span>
            </li>
          ))}
        </ol>

        <ul className="starters">
          {STARTERS.map((s) => (
            <li key={s.title}>
              <button type="button" onClick={() => onPick(s.text)}>
                <span>{s.title}</span>
                <i aria-hidden="true">→</i>
              </button>
            </li>
          ))}
        </ul>

        <p className="footnote">{FOOTNOTE}</p>
      </div>
    </section>
  );
}