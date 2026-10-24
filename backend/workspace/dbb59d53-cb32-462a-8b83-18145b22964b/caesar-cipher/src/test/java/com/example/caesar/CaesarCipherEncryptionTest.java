package com.example.caesar;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CaesarCipherEncryptionTest {

    @Test
    void shiftsLowercaseAndUppercase() {
        assertEquals("Khoor", CaesarCipherEncryption.encryption("Hello", 3));
    }

    @Test
    void wrapsAroundTheAlphabet() {
        assertEquals("abc", CaesarCipherEncryption.encryption("xyz", 3));
    }

    @Test
    void leavesNonLettersAlone() {
        assertEquals("Khoor, 123!", CaesarCipherEncryption.encryption("Hello, 123!", 3));
    }

    @Test
    void handlesNegativeShift() {
        assertEquals("xyz", CaesarCipherEncryption.encryption("abc", -3));
    }
}
