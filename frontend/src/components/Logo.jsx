import { APP_NAME, LOGO_URL } from "../brand";

export default function Logo({ size = 28 }) {
  return <img className="logo" src={LOGO_URL} width={size} height={size} alt={`${APP_NAME} logo`} />;
}
