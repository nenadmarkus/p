module Jekyll
    class ReplaceMathBlocksGenerator < Jekyll::Generator
      def generate(site)
        (site.pages + site.posts.docs).each do |document|
          document.content = replace_math_blocks(document.content)
        end
      end
  
      private
  
      def replace_math_blocks(content)
        # Replace $` ... `$ with $ ... $
        content.gsub(/\$\`\s*(.*?)\s*\`\$/, '$\1$')
      end
    end
  end
  