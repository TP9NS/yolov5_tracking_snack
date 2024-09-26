from flask import Flask, request, jsonify

app = Flask(__name__)

# 예시 상품 데이터베이스
products_db = {
    1: {"name": "abc_choco_cookie", "price": 1000, "image": "static/choco.jpg"},
    2: {"name": "chicchoc", "price": 500, "image": "static/chic.jpg"},
    3: {"name": "pocachip_original", "price": 800, "image": "static/poka.jpg"},
    4: {"name": "osatsu", "price": 1000, "image": "static/osa.jpg"},
    5: {"name": "turtle_chips", "price": 1500, "image": "static/turtle.jpg"}
}

@app.route('/product', methods=['POST'])
def get_product():
    data = request.get_json()
    product_id = data.get('product_id')

    if product_id in products_db:
        return jsonify(products_db[product_id])
    else:
        return jsonify({"error": "Product not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
