package com.example.caesar;

public class CaesarCipherEncryption {

    /**
     * Encrypts text with a Caesar cipher. Letters are shifted (wrapping around
     * the alphabet); non-letters are left unchanged.
     */
    public static String encryption(String password, int shift) {

        StringBuilder encryptedPassword = new StringBuilder();

        for (char character : password.toCharArray()) {

            if (Character.isLetter(character)) {

                char firstLetter = Character.isLowerCase(character) ? 'a' : 'A';

                int position = character - firstLetter;

                // floorMod keeps the result in 0-25 even when shift is negative
                int newPosition = Math.floorMod(position + shift, 26);

                char encryptedCharacter = (char) (newPosition + firstLetter);

                encryptedPassword.append(encryptedCharacter);

            } else {

                encryptedPassword.append(character);
            }
        }

        return encryptedPassword.toString();
    }
}
