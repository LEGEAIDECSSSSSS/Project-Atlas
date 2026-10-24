import { useCallback, useState } from "react";

/** useState that persists to localStorage. */
export function useLocalStorage(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored === null ? initial : JSON.parse(stored);
    } catch {
      return initial;
    }
  });

  const set = useCallback(
    (next) => {
      setValue(next);
      try {
        localStorage.setItem(key, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
    },
    [key],
  );

  return [value, set];
}
