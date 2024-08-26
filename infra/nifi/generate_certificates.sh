#!/bin/bash

CERTS_DIR=/opt/certs
KEYSTORE_PATH=$CERTS_DIR/keystore.jks
TRUSTSTORE_PATH=$CERTS_DIR/truststore.jks
PASSWORD=changeit

# creat a certificate Repo
mkdir -p $CERTS_DIR

# Generate keystore
if [ ! -f "$KEYSTORE_PATH" ]; then
    echo "Generating keystore..."
    keytool -genkeypair -alias nifi-cert -keyalg RSA -keysize 2048 -storetype JKS \
        -keystore $KEYSTORE_PATH -validity 365 -storepass $PASSWORD \
        -dname "CN=localhost, OU=NiFi, O=Company, L=City, S=State, C=Country"
else
    echo "Keystore already exists. Skipping generation."
fi

# Export certificat
if [ ! -f "$CERTS_DIR/nifi-cert.crt" ]; then
    echo "Exporting certificate from keystore..."
    keytool -export -alias nifi-cert -keystore $KEYSTORE_PATH -file $CERTS_DIR/nifi-cert.crt -storepass $PASSWORD
fi

# Generate truststore
if [ ! -f "$TRUSTSTORE_PATH" ]; then
    echo "Generating truststore..."
    keytool -import -alias nifi-cert -file $CERTS_DIR/nifi-cert.crt -keystore $TRUSTSTORE_PATH -storepass $PASSWORD -noprompt
else
    echo "Truststore already exists. Skipping generation."
fi

echo "Keystore and Truststore generation completed."
