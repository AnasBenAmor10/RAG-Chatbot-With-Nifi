# Use the base NiFi image
FROM apache/nifi:2.0.0-M4

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/jdk-21.0.3-bellsoft-x86_64
ENV PATH="/opt/conda/bin:${PATH}"

# Switch to root user
USER root

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    maven \
    wget \
    tar \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    uuid-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for Python downloads
RUN mkdir /python_downloads

# Copy Python 3.11.9 source tarball into the container
COPY Python-3.11.9.tgz /python_downloads/Python-3.11.9.tgz

# Extract and install Python 3.11.9
RUN tar xvf /python_downloads/Python-3.11.9.tgz -C /python_downloads && \
    cd /python_downloads/Python-3.11.9 && \
    ./configure --enable-optimizations && \
    make && \
    make install && \
    cd / && \
    rm -rf /python_downloads/Python-3.11.9* && \
    apt-get clean

# Install required Python package
RUN pip3 install --upgrade pip
# Copy your Python dependencies file to the Docker image
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install -r /tmp/requirements.txt

# Modify NiFi properties to use Python 3.11
RUN sed -i '/^nifi.python.command=/s/^#* *//g' /opt/nifi/nifi-current/conf/nifi.properties && \
    sed -i 's|nifi.python.command=.*|nifi.python.command=/usr/local/bin/python3.11|' /opt/nifi/nifi-current/conf/nifi.properties && \
    sed -i 's|nifi.python.working.directory=.*|nifi.python.working.directory=./work/python|' /opt/nifi/nifi-current/conf/nifi.properties && \
    sed -i 's|nifi.python.extensions.source.directory.default=.*|nifi.python.extensions.source.directory.default=/opt/nifi/nifi-current/python_extensions|' /opt/nifi/nifi-current/conf/nifi.properties



RUN username="orthlane" && \
    password='$2b$12$b67Z8xG7H5yp14uTCrFGZe8UxEU8MivzpnKo4EK6tixcAFrqPeUfy' && \
    properties_file="/opt/nifi/nifi-current/conf/nifi.properties" && \
    grep -q '^nifi.security.user.single.user.username=' "$properties_file" && \
    sed -i "s|^nifi.security.user.single.user.username=.*|nifi.security.user.single.user.username=${username}|" "$properties_file" || \
    echo "nifi.security.user.single.user.username=${username}" >> "$properties_file" && \
    grep -q '^nifi.security.user.single.user.password=' "$properties_file" && \
    sed -i "s|^nifi.security.user.single.user.password=.*|nifi.security.user.single.user.password=${password}|" "$properties_file" || \
    echo "nifi.security.user.single.user.password=${password}" >> "$properties_file"

# Update the value of Username and Password
RUN sed -i 's|<property name="Username">.*</property>|<property name="Username">$username</property>|' /opt/nifi/nifi-current/conf/login-identity-providers.xml && \
    sed -i 's|<property name="Password">.*</property>|<property name="Password">$password</property>|' /opt/nifi/nifi-current/conf/login-identity-providers.xml
# Copy Jython standalone jar into NiFi lib directory
COPY jython-standalone-2.7.2.jar /opt/nifi/nifi-current/lib/


# Copy custom Python processors into NiFi directory
COPY  nifi-python-extensions/extracted_extensions/extensions /opt/nifi/nifi-current/python_extensions/

# Ensure correct permissions for the custom processors
RUN chmod -R 755 /opt/nifi/nifi-current/python_extensions && \
    chmod +x /opt/nifi/nifi-current/python_extensions/chunking && \
    chmod +x /opt/nifi/nifi-current/python_extensions/vectorstores && \
    chmod +x /opt/nifi/nifi-current/python_extensions/scrapping && \
    chmod +x /opt/nifi/nifi-current/python_extensions/embeddings
# Expose necessary ports (adjust as per your NiFi setup)
EXPOSE 8080 8443



# Start NiFi
CMD ["/opt/nifi/nifi-current/bin/nifi.sh", "run"]
