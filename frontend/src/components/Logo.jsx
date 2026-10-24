import { APP_NAME } from "../brand";

// Two meridians meeting at the pole: a hairline on the left, a heavier cut on the right,
// with a single latitude for a crossbar. It inherits its colour (and the accent) from CSS.
export default function Logo({ size = 28 }) {
  return (
    <svg
      className="logo"
      width={size}
      height={size}
      viewBox="8 6 48 52"
      fill="none"
      role="img"
      aria-label={`${APP_NAME} logo`}
    >
      <path d="M32 9C44 20 49.5 38 48 56h-7.4C41.6 40 38 24 32 14Z" fill="currentColor" />
      <path d="M32 10C20 21 15.5 38 17 56" stroke="currentColor" strokeWidth="2.4" />
      <path d="M18.6 44.3Q30 41.8 41.4 44.3" stroke="var(--accent)" strokeWidth="2.6" />
    </svg>
  );
}