import concurrent.futures
import inspect
import os
import time
import traceback

# grpc c-ares doesn't resolve LAN/mDNS names
os.putenv("GRPC_DNS_RESOLVER", "native")

import grpc
import msgpack

# 30MB is enough for 9MP of uncompressed images
MAX_MESSAGE_LENGTH = 2 ** 20 * 30

BASE_OPTIONS = [
    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
]

class PogoServer:
    def __init__(self, obj, port=6869, creds=None):
        objname = obj.__class__.__name__

        handlers = {}
        directory = {}

        def wrap_method(meth):
            def wrapper(args, context):
                try:
                    return meth(args)
                except Exception as e:
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(traceback.format_exc())
                    return {'error': str(e)}
            return grpc.unary_unary_rpc_method_handler(wrapper,
                request_deserializer=msgpack.unpackb,
                response_serializer=msgpack.packb)

        for name, method in inspect.getmembers(obj, predicate=inspect.ismethod):
            if name.startswith('_'):
                continue
            handlers[name] = wrap_method(method)
            directory[name] = method.__doc__

        handlers['_GetMethods'] = wrap_method(lambda arg: directory)

        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(),
            handlers=[grpc.method_handlers_generic_handler(objname, handlers)],
            options=BASE_OPTIONS)

        self.port = port

        if creds:
            server_key, server_cert, root_ca = creds
            server_creds = grpc.ssl_server_credentials(
                [(server_key, server_cert)],
                root_certificates=root_ca,
                require_client_auth=True)

            self.server.add_secure_port(f"[::]:{port}", server_creds)
        else:
            self.server.add_insecure_port(f"[::]:{port}")

    def start(self):
        print("PogoRPC listening on port", self.port)
        return self.server.start()

    def stop(self, grace=180):
        return self.server.stop(grace).wait()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.stop(30)


class PogoClient:
    def __init__(self, addr, servicename, creds=None):
        if creds:
            cn, client_key, client_cert, root_ca = creds
            creds = grpc.ssl_channel_credentials(
                private_key=client_key,
                certificate_chain=client_cert,
                root_certificates=root_ca,
            )
            self.channel = grpc.secure_channel(addr, creds,
                options=BASE_OPTIONS + [('grpc.ssl_target_name_override', cn)])
        else:
            self.channel = grpc.insecure_channel(addr, options=BASE_OPTIONS)

        self.directory = self.channel.unary_unary(f'/{servicename}/_GetMethods',
            request_serializer=msgpack.packb,
            response_deserializer=msgpack.unpackb)(b"", timeout=300)

        for name in self.directory:
            setattr(self, name, self.channel.unary_unary(f'/{servicename}/{name}',
                request_serializer=msgpack.packb,
                response_deserializer=msgpack.unpackb))
