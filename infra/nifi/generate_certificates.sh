#!/bin/bash

CERTS_DIR=/opt/certs/
KEYSTORE_PATH=$CERTS_DIR/keystore.jks
TRUSTSTORE_PATH=$CERTS_DIR/truststore.jks
KEYSTORE_PASSWORD=QKZv1hSWAFQYZ+WU1jjF5ank+l4igeOfQRp+OSbkkrs
TRUSTSTORE_PASSWORD=rHkWR1gDNW3R9hgbeRsT3OM3Ue0zwGtQqcFKJD2EXWE

# Créer un répertoire pour les certificats
mkdir -p $CERTS_DIR

echo "Directory content before generating keystore:"
ls -l $CERTS_DIR

# Générer le keystore
if [ ! -f "$KEYSTORE_PATH" ]; then
    echo "Génération du keystore..."
    keytool -genkeypair -alias nifi-cert -keyalg RSA -keysize 2048 -storetype JKS \
        -keystore $KEYSTORE_PATH -validity 365 -storepass $KEYSTORE_PASSWORD \
        -dname "CN=localhost, OU=NiFi, O=Company, L=City, S=State, C=Country" \
        -keypass $KEYSTORE_PASSWORD

    if [ $? -ne 0 ]; then
        echo "Erreur : Échec de la génération du keystore."
        exit 1
    fi
else
    echo "Keystore déjà existant. Génération ignorée."
fi

echo "Directory content after generating keystore:"
ls -l $CERTS_DIR

# Exporter le certificat du keystore
if [ ! -f "$CERTS_DIR/nifi-cert.crt" ]; then
    echo "Exportation du certificat depuis le keystore..."
    keytool -export -alias nifi-cert -keystore $KEYSTORE_PATH -file $CERTS_DIR/nifi-cert.crt -storepass $KEYSTORE_PASSWORD

    if [ $? -ne 0 ]; then
        echo "Erreur : Échec de l'exportation du certificat depuis le keystore."
        exit 1
    fi
else
    echo "Certificat déjà exporté. Exportation ignorée."
fi

# Générer le truststore
if [ ! -f "$TRUSTSTORE_PATH" ]; then
    echo "Génération du truststore..."
    keytool -import -alias nifi-cert -file $CERTS_DIR/nifi-cert.crt -keystore $TRUSTSTORE_PATH -storepass $TRUSTSTORE_PASSWORD -noprompt

    if [ $? -ne 0 ]; then
        echo "Erreur : Échec de la génération du truststore."
        exit 1
    fi
else
    echo "Truststore déjà existant. Génération ignorée."
fi

echo "Génération du keystore et du truststore terminée avec succès."
echo "Directory content after generating truststore:"
ls -l $CERTS_DIR
